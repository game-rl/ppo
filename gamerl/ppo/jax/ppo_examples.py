import os

import gymnasium as gym
import jax
import jax.numpy as jnp
from jax.nn.initializers import orthogonal, zeros
import jax.example_libraries.stax as stax
import jax.example_libraries.optimizers as optim
import numpy as np
import matplotlib.pyplot as plt

from ppo import PPOTrainer

# This file shows how to train a PPO Agent on several gym environments.
# The agent uses separate policy and value networks and Adam optimizer.
# The hyperparameters for the training **mostly** follow StableBaselines3:
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml


# EnvironmentStepFn is a Callable that steps the environment.
class EnvironmentStepFn:
    def __init__(self, env):
        self.env = env
        self.o, _ = env.reset(seed=0)
    def __call__(self, acts):
        if acts is None:
            return self.o, None, None, None, None
        acts = np.asarray(acts)
        res = self.env.step(acts)
        self.o = res[0]
        return res

# ActorCritic is a Callable that selects actions for the given observations.
class ActorCritic:
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic
    def __call__(self, rng, params, obs, acts=None):
        pi_params, vf_params = params
        logits = self.actor(pi_params, obs)
        if acts is None:
            acts = jax.random.categorical(rng, logits)
        one_hot = jax.nn.one_hot(acts, num_classes=logits.shape[1])
        logp = one_hot * jax.nn.log_softmax(logits, axis=-1)
        logp = jnp.sum(logp, axis=-1)
        vals = self.critic(vf_params, obs)
        vals = jnp.squeeze(vals, axis=-1)
        # entropy = _entropy(logits)
        return acts, logp, vals

# def _entropy(logits):
#     min_real = 1e-8 # torch.finfo(self.logits.dtype).min
#     logits = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True) # normalize
#     logits = jnp.clip(logits, a_min=min_real)
#     probs = jax.nn.softmax(logits, axis=-1)
#     p_log_p = logits * probs
#     return -p_log_p.sum(axis=-1)

# OptimizerFn is a Callable that updates the parameters.
class OptimizerFn:
    def __init__(self, opt_update, get_params):
        self.opt_update = opt_update
        self.get_params = get_params
        self.step = 0
    def __call__(self, params, grads, opt_state):
        max_norm = 1.
        leaves, _ = jax.tree.flatten(grads)
        total_norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef = jnp.maximum(clip_coef, 1.0)
        grads = jax.tree.map(lambda g: clip_coef * g, grads) # clip grads
        opt_state = self.opt_update(self.step, grads, opt_state)
        params = self.get_params(opt_state)
        self.step += 1
        return params, opt_state

# CartPole trains a PPO Agent on the CartPole-v1 gym environment.
def CartPole():
    np.random.seed(seed=0)
    rng = jax.random.PRNGKey(seed=0)

    # Define the EnvironmentStepFn.
    num_envs = 8
    steps_limit = 500
    env = gym.wrappers.vector.RecordEpisodeStatistics(
        gym.vector.SyncVectorEnv([
            lambda: gym.make("CartPole-v1", max_episode_steps=steps_limit)
            for _ in range(num_envs)
        ]),
    )
    env_fn = EnvironmentStepFn(env)

    # Define the ActorCritic function.
    in_shape = env.single_observation_space.shape
    out_size = env.single_action_space.n
    pi_init_fn, pi_apply_fn = stax.serial(  # Policy network [in, hid=32, out]
        stax.Dense(out_dim=32, W_init=orthogonal(scale=np.sqrt(2)), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=out_size, W_init=orthogonal(scale=0.01), b_init=zeros),
    )
    vf_init_fn, vf_apply_fn = stax.serial(  # Value network [in, hid=32, out=1]
        stax.Dense(out_dim=32, W_init=orthogonal(scale=np.sqrt(2)), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=1, W_init=orthogonal(scale=1.), b_init=zeros),
    )
    rng, pi_rng, vf_rng = jax.random.split(rng, num=3)
    _, pi_params = pi_init_fn(pi_rng, (-1,) + in_shape)
    _, vf_params = vf_init_fn(vf_rng, (-1,) + in_shape)
    params = (pi_params, vf_params)
    agent_fn = ActorCritic(jax.jit(pi_apply_fn), jax.jit(vf_apply_fn))

    # Define the OptimizerFn.
    opt_init, opt_update, get_params = optim.adam(step_size=1e-3)
    opt_state = opt_init(params)
    optim_fn = OptimizerFn(jax.jit(opt_update), get_params)

    # Define the PPO Trainer.
    ppo_trainer = PPOTrainer(
        agent_fn, optim_fn, env_fn,
        ent_coef=0., vf_clip=float("inf"), discount=0.98, lamb=0.8, n_epochs=20, batch_size=256, tgt_KL=0.02,
    )

    log_dir = os.path.join("logs", "CartPole-v1")
    os.makedirs(log_dir, exist_ok=True)

    # Run the trainer and plot the results.
    # The total number of time steps is ``num_itres x time_steps x num_envs``.
    #                              2.5e5 =  1000    x    32      x   8
    # Instead of running once, we will run the trainer k times consecutively, and
    # we will record a demo after every training session.
    num_iters, time_steps = 1000, 32
    k = 4
    rng, rng_ = jax.random.split(rng, num=2)
    record_demo(rng_, log_dir, "run_0", agent_fn, params, "CartPole-v1") # record the initial agent
    for i in range(k):
        rng, rng2, rng3 = jax.random.split(rng, num=3)
        params, opt_state = ppo_trainer(rng2, params, opt_state, num_iters // k, time_steps)
        record_demo(rng3, log_dir, f"run_{i+1}", agent_fn, params, "CartPole-v1")
    generate_plots(log_dir, ppo_trainer.train_log, num_iters*time_steps*num_envs)
    env.close()

# LunarLander trains a PPO Agent on the LunarLander-v3 gym environment.
def LunarLander():
    np.random.seed(seed=0)
    rng = jax.random.PRNGKey(seed=0)

    # Define the EnvironmentStepFn.
    num_envs = 16
    steps_limit = 500
    env = gym.wrappers.vector.RecordEpisodeStatistics(
        gym.vector.SyncVectorEnv([
            lambda: gym.make(
                "LunarLander-v3",
                gravity=np.clip(np.random.normal(-10.0, 1.0), -11.99, -0.01),
                enable_wind=np.random.choice([True, False]),
                wind_power=np.clip(np.random.normal(15.0, 1.0), 0.01, 19.99),
                turbulence_power=np.clip(np.random.normal(1.5, 0.5), 0.01, 1.99),
                max_episode_steps=steps_limit,
            )
            for _ in range(num_envs)
        ]),
    )
    env_fn = EnvironmentStepFn(env)

    # Define the ActorCritic function.
    in_shape = env.single_observation_space.shape
    out_size = env.single_action_space.n
    pi_init_fn, pi_apply_fn = stax.serial(  # Policy network [in, hid=[64, 64], out]
        stax.Dense(out_dim=64, W_init=orthogonal(scale=np.sqrt(2)), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=64, W_init=orthogonal(scale=np.sqrt(2)), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=out_size, W_init=orthogonal(scale=0.01), b_init=zeros),
    )
    vf_init_fn, vf_apply_fn = stax.serial(  # Value network [in, hid=[64, 64], out=1]
        stax.Dense(out_dim=64, W_init=orthogonal(scale=np.sqrt(2)), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=64, W_init=orthogonal(scale=np.sqrt(2)), b_init=zeros),
        stax.Tanh,
        stax.Dense(out_dim=1, W_init=orthogonal(scale=1.), b_init=zeros),
    )
    rng, pi_rng, vf_rng = jax.random.split(rng, num=3)
    _, pi_params = pi_init_fn(pi_rng, (-1,) + in_shape)
    _, vf_params = vf_init_fn(vf_rng, (-1,) + in_shape)
    params = (pi_params, vf_params)
    agent_fn = ActorCritic(jax.jit(pi_apply_fn), jax.jit(vf_apply_fn))

    # Define the OptimizerFn.
    opt_init, opt_update, get_params = optim.adam(step_size=3e-4)
    opt_state = opt_init(params)
    optim_fn = OptimizerFn(jax.jit(opt_update), get_params)

    # Define the PPO Trainer.
    ppo_trainer = PPOTrainer(
        agent_fn, optim_fn, env_fn,
        ent_coef=0.01, discount=0.999, lamb=0.98, n_epochs=4, batch_size=64, tgt_KL=None,
    )

    log_dir = os.path.join("logs", "LunarLander-v3")
    os.makedirs(log_dir, exist_ok=True)

    # Run the trainer and plot the results.
    # The total number of time steps is ``num_itres x time_steps x num_envs``.
    #                              2.5e6 =   300    x    512     x   16
    # Instead of running once, we will run the trainer k times consecutively, and
    # we will record a demo after every training session.
    num_iters, time_steps = 300, 512
    k = 4
    rng, rng_ = jax.random.split(rng, num=2)
    record_demo(rng_, log_dir, "run_0", agent_fn, params, "LunarLander-v3") # record the initial agent
    for i in range(k):
        rng, rng2, rng3 = jax.random.split(rng, num=3)
        params, opt_state = ppo_trainer(rng2, params, opt_state, num_iters // k, time_steps)
        record_demo(rng3, log_dir, f"run_{i+1}", agent_fn, params, "LunarLander-v3")
    generate_plots(log_dir, ppo_trainer.train_log, num_iters*time_steps*num_envs)
    env.close()

def record_demo(rng, log_dir, video_name, agent_fn, params, env_name):
    env = gym.wrappers.RecordVideo(
        gym.wrappers.Autoreset(
            gym.make(env_name, render_mode="rgb_array"),
        ),
        video_folder=log_dir,
        video_length=1000, # around 20 sec, depends on fps (usually 50fps)
        name_prefix=video_name,
    )
    seed = jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max).item()
    o, _ = env.reset(seed=seed)
    while env.recording:
        o = jnp.asarray(np.expand_dims(o, axis=0))
        rng, rng_ = jax.random.split(rng, num=2)
        acts, _, _ = agent_fn(rng_, params, o)
        acts = np.asarray(acts)[0]
        o, r, t, tr, info = env.step(acts)
    env.close()

def generate_plots(log_dir, train_log, total_steps):
    plt.style.use("ggplot")

    keys = (
        "Policy Loss", "Value Loss", "Total Loss", "Entropy Bonus",
        "KL Divergence", "Total Grad Norm",
    )
    for k in keys:
        if not isinstance(train_log[k][0], float):
            continue
        fig, ax = plt.subplots()
        ax.plot(train_log[k], label=k)
        ax.legend()
        ax.set_xlabel("Gradient updates")
        ax.set_ylabel(k)
        fig.savefig(os.path.join(log_dir, k.replace(" ", "_")+".png"))

    keys = ("Episode Returns", "Episode Lengths")
    for k in keys:
        if not isinstance(train_log[k][0], tuple) or len(train_log[k][0]) != 3:
            continue
        num_records = len(train_log[k])
        avg, std, run = zip(*train_log[k])
        avg, std, run = np.array(avg), np.array(std), np.array(run)
        # xs = np.cumsum(list([0]+train_log["Num Updates"])[:-1])
        xs = np.linspace(0, total_steps, num_records)
        xs_ = xs[~(avg != avg)]     # Remove NaNs, if any.
        avg = avg[~(avg != avg)]
        std = std[~(std != std)]
        fig, ax = plt.subplots()
        ax.plot(xs_, avg, label="Average")
        ax.fill_between(xs_, avg-0.5*std, avg+0.5*std, color="k", alpha=0.25)
        ax.plot(xs, run, label="Running")
        ax.legend()
        ax.set_xlabel("Number of time-steps")
        ax.set_ylabel(k)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0,0))
        fig.savefig(os.path.join(log_dir, k.replace(" ", "_")+".png"))


if __name__ == "__main__":
    CartPole()
    LunarLander()

    # # Manual play
    # from gymnasium.utils.play import play
    # play(
    #     gym.make("LunarLander-v2", render_mode="rgb_array"),
    #     keys_to_action={"w":1, "a":2, "s":3, "d":4},
    # )

#