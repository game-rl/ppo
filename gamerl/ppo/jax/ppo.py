from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Sequence
import warnings

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from tqdm import tqdm


Key = Any       # Key for a pseudo random number generator (PRNG)
PyTree = Any    # PyTrees are arbitrary nests of ``jnp.ndarrays``
OptState = Any  # Arbitrary object holding the state of the optimizer.

# ActorCritic func takes the model parameters and the observations
# as input, and returns: the actions for each observation; the log
# probabilities for each of the selected actions; the value
# estimates for each observation. The function also accepts the
# actions for each observation as an optional input and returns
# the same actions and their log probabilities.
ActorCritic = Callable[
    [Key, PyTree, ArrayLike, Optional[ArrayLike]],
    tuple[jax.Array, jax.Array, jax.Array],
]

# OptimizerFn takes parameters, their gradients, and the
# optimizer state as input and returns the updated parameters and
# the new state.
OptimizerFn = Callable[[PyTree, PyTree, OptState], tuple[PyTree, OptState]]

# EnvironmentStepFn is a step function for a vectorized environment
# conforming to the Gymnasium environments API. See:
#   https://gymnasium.farama.org/api/env/#gymnasium.Env.step
#   https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv.step
#
# If ``None`` is given as input, then the function returns the
# observations for the current state of the environment.
EnvironmentStepFn = Callable[
    [ArrayLike | None],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict],
]


@dataclass
class PPOTrainer:
    """
        ``ppo_trainer = PPOTrainer(**kwargs)``

        ``params, opt_state = ppo_trainer(rng, params, opt_state, num_iters, num_steps)``
    """
    agent_fn: ActorCritic
    optim_fn: OptimizerFn
    env_fn: EnvironmentStepFn
    pi_clip: float = 0.2
    vf_clip: float = 1.
    vf_coef: float = 0.5
    ent_coef: float = 0.
    tgt_KL: float = 0.02
    discount: float = 1.
    lamb: float = 0.95
    n_epochs: int = 5
    batch_size: int = 128
    run_ret: float = field(default=np.nan, init=False)
    run_len: float = field(default=np.nan, init=False)
    train_log: dict[str, list[Any]] = field(default_factory=lambda: defaultdict(list), init=False)

    def __call__(
        self,
        rng: Key,
        params: PyTree,
        opt_state: OptState,
        N: int,
        T: int,
    ) -> tuple[PyTree, OptState]:
        """Run the PPO trainer for multiple iterations to update the model
        parameters. Each iteration consists of two stages:\n
          i) data collection stage, where sample trajectories of fixed length
          are generated using the current policy;\n
          ii) parameter optimization stage, where multiple update steps are
          applied to optimize the PPO-CLIP objective.

        Args:
            rng: Key
                A PRNG key array.
            params: PyTree
                Current model parameters for the agent function.
            opt_state: OptState
                Current optimizer state for the optimizer function.
            N: int
                Number of iterations for running the trainer.
            T: int
                Number of steps for generating fixed-length trajectories.

        Returns:
            PyTree
                The updated model parameters.
            OptState
                The latest state of the optimizer.
        """
        for _ in tqdm(range(N)):
            # Rollout the environment.
            rng, rng_ = jax.random.split(rng, num=2)
            obs, acts, rewards, done, logp, values, info = \
                environment_loop(rng_, self.env_fn, self.agent_fn, params, T)

            # Compute the generalized advantages.
            adv = gae(values, rewards, done, self.discount, self.lamb)

            # Reshape the arrays. Note that we are reshaping only after the
            # advantages have been computed.
            dataset = (
                obs.reshape(-1, *obs.shape[2:]), acts.ravel(), adv.ravel(), logp.ravel(), values.ravel(),
            )

            # Iterate over the dataset by sampling mini-batches and update the
            # parameters of the model.
            n_updates = 0
            stop_training = False
            for _ in range(self.n_epochs):
                rng, rng_ = jax.random.split(rng, num=2)
                loader = data_loader(rng_, dataset, self.batch_size)

                for o, a, ad, lp, vals in loader:
                    # Compute the clipped ppo loss and the gradients of the params.
                    rng, rng_ = jax.random.split(rng, num=2)
                    (loss, aux), grads = ppo_clip_loss(
                        rng_, self.agent_fn, params, o, a, ad, lp, vals,
                        self.pi_clip, self.vf_clip, self.vf_coef, self.ent_coef,
                    )

                    # Bookkeeping.
                    pi_loss, vf_loss, s_ent, kl_div = aux
                    leaves, _ = jax.tree.flatten(grads)
                    grad_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
                    self.train_log["Total Grad Norm"].append(grad_norm.item())
                    self.train_log["Total Loss"].append(loss.item())
                    self.train_log["Policy Loss"].append(pi_loss.item())
                    self.train_log["Value Loss"].append(vf_loss.item())
                    self.train_log["Entropy Bonus"].append(s_ent.item())
                    self.train_log["KL Divergence"].append(kl_div.item())

                    # Backward pass. Update the parameters of the model.
                    params, opt_state = self.optim_fn(params, grads, opt_state)
                    n_updates += 1

                    # Check for early stopping. If the mean KL-divergence of the
                    # new policy from the old grows beyond the threshold, we
                    # stop taking gradient steps.
                    if self.tgt_KL is not None and kl_div > 1.5 * self.tgt_KL:
                        stop_training = True
                        break

                # Stop the ppo training early and collect new data.
                if stop_training:
                    break

            # Bookkeeping.
            for r in info["episode_returns"]:
                self.run_ret = r if self.run_ret is np.nan else 0.99 * self.run_ret + 0.01 * r
            for l in info["episode_lengths"]:
                self.run_len = l if self.run_len is np.nan else 0.99 * self.run_len + 0.01 * l
            with warnings.catch_warnings():
                # We might finish the rollout without completing any episodes. In
                # this case we want to store NaN in the history. Taking the mean or
                # std of an empty slice throws a runtime warning and returns a NaN,
                # which is exactly what we want.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_r, std_r = np.mean(info["episode_returns"]), np.std(info["episode_returns"])
                avg_l, std_l = np.mean(info["episode_lengths"]), np.std(info["episode_lengths"])
            self.train_log["Num Updates"].append(n_updates)
            self.train_log["Episode Returns"].append((avg_r, std_r, self.run_ret))
            self.train_log["Episode Lengths"].append((avg_l, std_l, self.run_len))

        return (params, opt_state)


# Differentiate only the first output of the function, and threat
# the second output as auxiliary data. Differentiation is done
# with respect to the third input parameter, i.e. model params.
@partial(jax.jit, static_argnames="agent_fn")
@partial(jax.value_and_grad, argnums=2, has_aux=True)
def ppo_clip_loss(
    rng: Key,
    agent_fn: ActorCritic,
    params: PyTree,
    obs: ArrayLike,
    acts: ArrayLike,
    adv: ArrayLike,
    logp_old: ArrayLike,
    v_old: ArrayLike,
    pi_clip: float,
    vf_clip: float,
    c1: float,
    c2: float,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Compute the PPO-CLIP loss.
    See: https://arxiv.org/abs/1707.06347

        L_CLIP = L_pi - c_1 L_vf + c_2 S_ent

    Args:
        rng: Key
            A PRNG key array.
        agent_fn: ActorCritic
            Actor-critic agent function.
        params: PyTree
            The parameters of the model.
        obs: ArrayLike
            Array of shape (B, *) giving a batch of observations.
        acts: ArrayLike
            Array of shape (B,) giving the selected actions.
        adv: ArrayLike
            Array of shape (B,) giving the computed advantages
            for each (obs, act) pair.
        logp_old: ArrayLike
            Array of shape (B,) giving the log probs for each action.
        v_old: ArrayLike
            Array of shape (B,) giving the values for each obs.
        pi_clip: float
            Clip ratio for clipping the policy objective.
        vf_clip: float
            Clip value for clipping the value objective.
        c1: float
            Factor for augmenting the loss with the value loss.
        c2: float
            Factor for augmenting the loss with the entropy bonus.

    Returns:
        jax.Array
            Array of size 1. holding the value of the ppo-clip loss.
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]
            A tuple holding the values of the policy loss, value
            loss, the mean policy entropy and the mean KL-divergence.
    """
    # Convert the inputs to jax arrays.
    obs = jnp.asarray(obs)
    adv = jnp.asarray(adv)
    logp_old = jnp.asarray(logp_old)
    v_old = jnp.asarray(v_old)

    # Compute the TD(λ)-returns.
    v_old = jax.lax.stop_gradient(v_old) # old values with no gradient
    returns = adv + v_old

    # Normalize the advantages at the mini-batch level.
    # This operation can be viewed as applying a state-dependent
    # baseline and scaling the learning rate.
    eps = jnp.finfo(adv.dtype).eps
    adv = (adv - adv.mean()) / (adv.std() + eps)

    # Compute the clipped policy loss.
    logp_old = jax.lax.stop_gradient(logp_old)         # old log probs with no gradient
    _, logp, v_pred = agent_fn(rng, params, obs, acts) # new log probs
    rho = jnp.exp(logp - logp_old)
    clip_adv = jnp.clip(rho, 1-pi_clip, 1+pi_clip) * adv
    pi_loss = jnp.mean(jnp.minimum(rho * adv, clip_adv))

    # Calculate the clipped value loss.
    v_clip = v_old + jnp.clip(v_pred-v_old, -vf_clip, vf_clip)
    vf_loss = jnp.mean(jnp.maximum((v_pred - returns)**2, (v_clip - returns)**2))

    # Approximate the policy entropy: ``S_ent = -E_p [ log p(x) ]``.
    # Note that in most cases we can compute the entropy exactly using the logits.
    # This, however, is not needed as this approximation is good enough for entropy bonus.
    s_ent = -jnp.mean(logp)

    # Compute the total loss.
    total_loss = -(pi_loss - c1 * vf_loss + c2 * s_ent)

    # Approximate the KL-divergence between the old and the new policies:
    # For details see: http://joschu.net/blog/kl-approx.html
    logr = logp - logp_old
    kl_div = jnp.mean(jnp.exp(logr) - 1 - logr)

    return total_loss, (pi_loss, vf_loss, s_ent, kl_div)


@jax.jit
def gae(
    values: ArrayLike,
    rewards: ArrayLike,
    done: ArrayLike,
    gamma: float,
    lamb: float,
) -> jax.Array:
    """Compute the generalized advantage estimations.
    See: https://arxiv.org/abs/1506.02438

    Args:
        values: ArrayLike
            Array of shape (T,) or (T, B) with the computed values.
        rewards: ArrayLike
            Array of shape (T,) or (T, B) with the obtained rewards.
        done: ArrayLike
            Boolean array of shape (T,) or (T, B) indicating which
            steps are terminal steps for the environment.
        gamma: float
            Discount factor for future rewards.
        lamb: float
            Weighting factor for n-step updates. (Similar to TD(λ))

    Returns:
        jax.Array
            Array of shape (T,) or (T, B) with the generalized
            advantage estimations for each time step.
    """
    # Convert the inputs to jax arrays.
    values = jnp.asarray(values)
    rewards = jnp.asarray(rewards)
    done = jnp.asarray(done, dtype=bool)

    # Stop gradients for the values.
    values = jax.lax.stop_gradient(values)

    T = values.shape[0] # number of time-steps

    # For unfinished episodes we will bootstrap the last reward:
    #   ``r_T = r_T + V(s_T), if s_T not terminal``
    adv = jnp.where(done[-1], rewards[-1] - values[-1], rewards[-1]) # A = r - V
    result = [None] * T
    result[-1] = adv

    # Compute the advantages in reverse order.
    for t in range(T-2, -1, -1): # O(T)  \_("/)_/
        # TD-residual ``δ_t = r_t + γ V(s_{t+1}) - V(s_t)``
        delta = rewards[t] + gamma * values[t+1] * ~done[t] - values[t]

        # Generalized advantage ``A_GAE(t) = δ_t + γλ A_GAE(t+1)``
        adv = delta + lamb * gamma * adv * ~done[t]

        # Store the result.
        result[t] = adv

    return jnp.stack(result)


def environment_loop(
    rng: Key,
    env_fn: EnvironmentStepFn,
    agent_fn: ActorCritic,
    params: PyTree,
    T: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict]:
    """Rollout the agent policy by stepping the environment for ``T`` steps.

    Args:
        rng: Key
            A PRNG key array.
        env_fn: EnvironmentStepFn
            Function for stepping the environment given the actions.
        agent_fn: ActorCritic
            Agent function used for selecting actions.
        params: PyTree
            The parameters of the model.
        T: int
            Number of time-steps to step the environment.

    Returns:
        jax.Array
            Observations array of shape (T, B, *).
        jax.Array
            Actions array of shape (T, B).
        jax.Array
            Rewards array of shape (T, B).
        jax.Array
            Boolean array of shape (T, B) indicating which states are terminal.
        jax.Array
            Logprobs array of shape (T, B) giving the log probability for each
            of the actions.
        jax.Array
            Values array of shape (T, B) giving the value estimate for the states.
        dict[str, Sequence[float]]
            Info dict.
    """
    episode_returns, episode_lengths = [], []

    # Allocate containers for the observations during rollout.
    obs, actions, rewards, done, logprobs, values = \
        [None] * T, [None] * T, [None] * T, [None] * T, [None] * T, [None] * T

    # Observe the current state of the environment by passing ``None`` for acts.
    o, *_ = env_fn(None) # shape (B, *)
    B = o.shape[0]

    for i in range(T):
        obs[i] = o

        # Run the current obs through the agent network and step
        # the environment with the selected actions.
        rng, acts_rng = jax.random.split(rng, num=2)
        acts, logp, vals = agent_fn(acts_rng, params, o)
        o, r, t, tr, infos = env_fn(acts)

        # TODO:
        # # If some environment was truncated, then extract the final obs from
        # # the returned info.
        # if tr.any():
        #     o_next = np.zeros_like(o)
        #     for k in range(num_envs):
        #         next_obs[k] = o[k] if not tr[k] else infos["final_observation"][k]
        #     rng, vals_rng = jax.random.split(rng, num=2)
        #     _, _, v_next = agent_fn(vals_rng, params, o_next)
        #     r = jnp.where(tr, r + v_next, r)

        # If any of the environments is done, then save the statistics.
        if (t | tr).any():
            episode_returns.extend([
                infos["episode"]["r"][k] for k in range(B) if (t | tr)[k]
            ])
            episode_lengths.extend([
                infos["episode"]["l"][k] for k in range(B) if (t | tr)[k]
            ])

        # Store the observations and the agent output.
        actions[i] = acts
        rewards[i] = r
        done[i] = (t | tr)
        logprobs[i] = logp
        values[i] = vals

    # Stack the experiences into arrays of shape (T, B, *), where
    # T is the number of steps and B is the number of sub-envs.
    obs = jnp.stack(obs)
    actions = jnp.stack(actions)
    rewards = jnp.stack(rewards)
    done = jnp.stack(done, dtype=bool)
    logprobs = jnp.stack(logprobs)
    values = jnp.stack(values)

    info = {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
    }

    return (obs, actions, rewards, done, logprobs, values, info)


def data_loader(
    rng: Key,
    dataset: Sequence[ArrayLike],
    batch_size: int,
) -> Iterable[list[jax.Array]]:
    """Iterate over the given dataset. The iterator samples random
    batches without replacement.

    Args:
        rng: Key
            A PRNG key array.
        dataset: Sequence[ArrayLike]
            The dataset consists of ArrayLike objects. Each array
            must have the same number of examples.
        batch_size: int
            Size of sampled batches.

    Returns:
        Iterable[list[jax.Array]]
            An iterator for mini-batches of examples from the dataset.
    """
    num_train = dataset[0].shape[0]
    it = jax.random.permutation(rng, jnp.arange(num_train)).tolist()
    it = iter(it)
    while True:
        # Sample without replacement a random batch of indices.
        try:
            # TODO: could we speed this up?
            # Instead of calling ``next()`` batch_size times, we
            # should be able to make a single call. Maybe by
            # coding custom ``__iter__`` and ``__next__`` funcs?

            # Yield a mini-batch from each of the arrays in the dataset.
            idxs = [ next(it) for _ in range(batch_size) ]
            idxs = jnp.asarray(idxs)
            yield [ jnp.asarray(x[idxs]) for x in dataset ]

        except StopIteration:
            return

#