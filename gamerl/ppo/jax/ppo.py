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


Key = Any
"""Key for a pseudo random number generator (PRNG)."""

PyTree = Any
"""PyTrees are arbitrary nests of `jnp.ndarrays`."""

OptState = Any
"""Arbitrary object holding the state of the optimizer."""

# ActorCritic func takes as input the model parameters and a batch
# of observations, and returns: the actions for each observation;
# the log probabilities for each of the selected actions; the
# value estimates for each observation. The function also accepts
# the actions for each observation as an optional input and
# returns the same actions and their log probabilities.
ActorCritic = Callable[
    [Key, PyTree, ArrayLike, Optional[ArrayLike]],
    tuple[jax.Array, jax.Array, jax.Array],
]

# OptimizerFn takes as input parameters, their gradients, and the
# optimizer state and returns the updated parameters and the new state.
OptimizerFn = Callable[[PyTree, PyTree, OptState], tuple[PyTree, OptState]]

# EnvironmentStepFn is a step function for a vectorized
# environment conforming to the Gymnasium environments API. See:
#   https://gymnasium.farama.org/api/env/#gymnasium.Env.step
#   https://gymnasium.farama.org/api/vector/#gymnasium.vector.VectorEnv.step
#
# The function takes as input a batch of actions to update the
# environment state, and returns the next observations and the
# rewards resulting from the actions. The function also returns
# boolean arrays indicating whether any of the sub-environments
# were terminated or truncated, as well as an info dict.
#
# If ``None`` is given as input, then the function returns the
# current observations without stepping the environment.
EnvironmentStepFn = Callable[
    [ArrayLike | None],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict],
]

# PPOTrainer is a callable that trains an agent to maximize
# expected return from a given environment.
@dataclass
class PPOTrainer:
    """PPOTrainer trains an actor-critic agent for multiple iterations.

    Each training iteration consists of two stages:
      1. Data collection stage. Fixed-length trajectories are collected
        using the current policy.
      2. Parameter optimization stage. The parameters of the model are
        optimized for the PPO-CLIP objective using multiple update steps.

    The agent and the environment are provided to the trainer upon
    initialization. The trainer can then be called multiple times to
    update the agent parameters. You can see how well the current agent
    is performing in between calling the trainer:

    ```python
    # Initialize the trainer.
    ppo_trainer = PPOTrainer(agent_fn, optim_fn, env_fn, **kwargs)

    # Call the trainer to update the current params.
    params, opt_state = ppo_trainer(rng1, params, opt_state, n_iters, n_steps, n_updates)

    # Test the agent after training.
    record_demo(env_fn, agent_fn, params)

    # Train the agent some more.
    params, opt_state = ppo_trainer(rng2, params, opt_state, n_iters, n_steps, n_updates)
    ```
    """

    agent_fn: ActorCritic
    optim_fn: OptimizerFn
    env_fn: EnvironmentStepFn
    pi_clip: float = 0.2    # clip ratio for the policy objective
    vf_clip: float = 1.     # clip ratio for the value objective
    vf_coef: float = 0.5    # value coef for loss calculation
    ent_coef: float = 0.    # entropy coef for loss calculation
    tgt_KL: float = 0.02    # target KL divergence for early stopping
    discount: float = 1.    # discount for future rewards
    lamb: float = 0.95      # lambda coef for GAE-lambda
    batch_size: int = 128   # batch size for iterating over observations
    train_log: dict[str, list[Any]] = field( # for logging useful info during training
        default_factory=lambda: defaultdict(list))

    def __call__(
        self,
        rng: Key,
        params: PyTree,
        opt_state: OptState,
        n_iters: int,
        n_steps: int,
        n_updates: int,
    ) -> tuple[PyTree, OptState]:
        """Update the model parameters using the PPO algorithm.

        Args:
            rng: Key
                PRNG key array.
            params: PyTree
                Current model parameters for the agent function.
            opt_state: OptState
                Current optimizer state for the optimizer function.
            n_iters: int
                Number of iterations for running the trainer.
            n_steps: int
                Number of steps for generating fixed-length trajectories.
            n_updates: int
                Number of ppo updates to be performed at every iteration.

        Returns:
            PyTree
                The updated model parameters.
            OptState
                The latest state of the optimizer.
        """

        self.train_log["hyperparams"].append({
            "n_iters": n_iters,
            "n_steps": n_steps,
            "n_updates": n_updates,
            "n_envs": self.env_fn(None)[0].shape[0]
        })

        for _ in tqdm(range(n_iters)):
            # Rollout the policy.
            rng, rng_ = jax.random.split(rng, num=2)
            (obs, acts, rewards, done, logp, values), info = \
                environment_loop(rng_, self.env_fn, self.agent_fn, params, n_steps)

            # Bookkeeping. Store episode stats averaged over the time-steps.
            with warnings.catch_warnings():
                # We might finish the rollout without completing episodes.
                # Just store NaNs and ignore the warning.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.train_log["ep_r"].append((np.mean(info["ep_r"]), np.std(info["ep_r"])))
                self.train_log["ep_l"].append((np.mean(info["ep_l"]), np.std(info["ep_l"])))

            # Compute the generalized advantages.
            adv = gae(values, rewards, done, self.discount, self.lamb)

            # Reshape the arrays and create a batch loader. Note that we
            # are reshaping only after the advantages have been computed.
            dataset = (
                obs.reshape(-1, *obs.shape[2:]), acts.ravel(), adv.ravel(),
                logp.ravel(), values.ravel(),
            )
            rng, rng_ = jax.random.split(rng, num=2)
            loader = data_loader(rng_, dataset, self.batch_size, n_updates)

            # Iterate over the dataset and update the parameters.
            losses, pi_losses, vf_losses, norms, pi_ents, kl_divs = [], [], [], [], [], []
            for u_, (o, a, ad, lp, vals) in enumerate(loader):
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
                losses.append(loss.item())
                pi_losses.append(pi_loss.item())
                vf_losses.append(vf_loss.item())
                norms.append(grad_norm.item())
                pi_ents.append(s_ent.item())
                kl_divs.append(kl_div.item())

                # Backward pass. Update the parameters of the model.
                params, opt_state = self.optim_fn(params, grads, opt_state)

                # Check for early stopping. If the mean KL-divergence
                # between the new policy and the old grows beyond the
                # threshold, we stop taking gradient steps.
                if self.tgt_KL is not None and kl_div > 1.5 * self.tgt_KL:
                    break

            # Bookkeeping. Store training stats averaged over the number of updates.
            self.train_log["n_updates"].append(u_) # actual number of updates
            self.train_log["loss"].append((np.mean(losses), np.std(losses)))
            self.train_log["pi_loss"].append((np.mean(pi_losses), np.std(pi_losses)))
            self.train_log["vf_loss"].append((np.mean(vf_losses), np.std(vf_losses)))
            self.train_log["grad_norm"].append((np.mean(norms), np.std(norms)))
            self.train_log["pi_ent"].append((np.mean(pi_ents), np.std(pi_ents)))
            self.train_log["kl_div"].append((np.mean(kl_divs), np.std(kl_divs)))

            # TODO: Add to the logger `clipfrac`: the fraction of
            # the training data that triggered the clipped objective.

        return (params, opt_state)

# Differentiate only the first output of the function, and treat
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

        ``L_CLIP = L_pi - c_1 L_vf + c_2 S_ent``

    Args:
        rng: Key
            PRNG key array.
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
            Array of size 1 holding the value of the ppo-clip loss.
        tuple[jax.Array, jax.Array, jax.Array, jax.Array]
            Tuple holding the values of the policy loss, value
            loss, the mean policy entropy and the mean KL-divergence.
    """

    obs = jnp.asarray(obs)
    adv = jnp.asarray(adv)
    logp_old = jnp.asarray(logp_old)
    v_old = jnp.asarray(v_old)

    # Compute the TD(λ)-returns.
    v_old = jax.lax.stop_gradient(v_old) # old values with no gradient
    returns = adv + v_old

    # Normalize the advantages at the mini-batch level. This
    # should help with stabilizing the training procedure.
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
    # Note that in most cases we can compute the entropy exactly using
    # the logits. This, however, is not needed as this approximation is
    # good enough for entropy bonus.
    s_ent = -jnp.mean(logp)

    # Compute the total loss.
    total_loss = -(pi_loss - c1 * vf_loss + c2 * s_ent)

    # Approximate the KL-divergence between the old and the new policies:
    # For details see: http://joschu.net/blog/kl-approx.html
    logr = logp - logp_old
    kl_div = jnp.mean(jnp.exp(logr) - 1 - logr)

    return total_loss, (pi_loss, vf_loss, s_ent, kl_div)

# gae computes generalized advantages and works both for batched
# and non-batched inputs so we can skip the @jax.vmap decorator.
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

    values = jnp.asarray(values)
    rewards = jnp.asarray(rewards)
    done = jnp.asarray(done, dtype=bool)

    # Stop gradients for the values, since here they will be used
    # for constructing the targets.
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

# environment_loop simulates the agent-environment interaction
# loop and returns the collected transitions.
def environment_loop(
    rng: Key,
    env_fn: EnvironmentStepFn,
    agent_fn: ActorCritic,
    params: PyTree,
    T: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, dict]:
    """Rollout the agent policy by stepping the environment for T steps.

    Args:
        rng: Key
            PRNG key array.
        env_fn: EnvironmentStepFn
            Function for stepping the environment given the actions.
        agent_fn: ActorCritic
            Agent function used for selecting actions.
        params: PyTree
            The parameters of the model.
        T: int
            Number of time-steps to step the environment.

    Returns:
        Transitions
            Tuple (obs, acts, rewards, done, logprobs, vals) of
            nd-arrays of shape (T, B, *) or (T, B), where B is the
            number of sub-environments.
        dict[str, Sequence[float]]
            Info dict.
    """

    o, *_ = env_fn(None) # shape (B, *)
    B = o.shape[0]
    ep_r, ep_l = [], []

    # Allocate containers for the observations during rollout.
    obs, actions, rewards, done, logprobs, values = \
        [None] * T, [None] * T, [None] * T, [None] * T, [None] * T, [None] * T

    for i in range(T):
        obs[i] = o

        # Run the current obs through the agent network and step
        # the environment with the selected actions.
        rng, rng_ = jax.random.split(rng, num=2)
        acts, logp, vals = agent_fn(rng_, params, o)
        o, r, t, tr, infos = env_fn(acts)

        # Store the observations and the agent output.
        actions[i] = acts
        rewards[i] = r
        done[i] = (t | tr)
        logprobs[i] = logp
        values[i] = vals

        # TODO: try to read the stats without using B. I.e. make
        # it work for both vectorized and non-vectorized envs.
        # Bookkeeping. On every step store episode lengths and returns.
        ep_r.extend((infos["episode"]["r"][k] for k in range(B) if (t|tr)[k]))
        ep_l.extend((infos["episode"]["l"][k] for k in range(B) if (t|tr)[k]))

    # Stack the experiences into arrays of shape (T, B, *), where
    # T is the number of steps and B is the number of sub-envs.
    obs = jnp.stack(obs)
    actions = jnp.stack(actions, dtype=int)
    rewards = jnp.stack(rewards)
    done = jnp.stack(done, dtype=bool)
    logprobs = jnp.stack(logprobs)
    values = jnp.stack(values)

    info = {"ep_r": ep_r, "ep_l": ep_l}

    return (obs, actions, rewards, done, logprobs, values), info

# data_loader iterates over the dataset yielding mini-batches.
def data_loader(
    rng: Key,
    dataset: Sequence[ArrayLike],
    batch_size: int,
    n_batches: int,
) -> Iterable[list[jax.Array]]:
    """Iterate over the dataset yielding a fixed number of batches.

    The iterator samples random batches without replacement. Once the
    dataset is exhausted, the iterators starts afresh with the entire
    dataset.

    Args:
        rng: Key
            PRNG key array.
        dataset: Sequence[ArrayLike]
            The dataset consists of ArrayLike objects. Each array
            must have the same number of examples.
        batch_size: int
            Size of sampled batches.
        n_batches: int
            The number of batches to be generated from the loader.

    Returns:
        Iterable[list[jax.Array]]
            Iterator for mini-batches of examples from the dataset.
    """

    size = dataset[0].shape[0]

    # Calculate the number of epochs that need to be performed in
    # order to produce n_batches. Note that ``num // den`` rounds
    # down, so use ``(num + den - 1) // den`` to round up.
    epochs = (n_batches * batch_size + size - 1) // size

    # For each epoch create a random permutation of the dataset
    # and stack the indices for continuous iteration.
    rng, *rngs = jax.random.split(rng, num=epochs+1)
    it = jnp.concat([
        jax.random.permutation(rngs[i], jnp.arange(size)) for i in range(epochs)
    ])

    # Iterate over the shuffled indices and yield mini-batches.
    for i in range(n_batches):
        idxs = it[i*batch_size:(i+1)*batch_size]
        yield [ jnp.asarray(x[idxs]) for x in dataset ]

#