
"""
A training script of Maximum a Posteriori Policy Optimisation on OpenAI Gym Mujoco environments.
This script follows the settings of https://arxiv.org/abs/1806.06920 as much
as possible.
"""
from distutils.version import LooseVersion
import functools
import logging
import os

import torch
from torch import nn
import gym
import gym.wrappers
import hydra
from omegaconf import DictConfig
import numpy as np

from pfrlx.algos import MPO
import pfrlx.experiments as experiments
import pfrlx.utils as utils
import pfrlx.replay_buffers as replay_buffers
import pfrlx.envs as envs
import pfrlx.wrappers as wrappers
import pfrlx.networks as networks
import pfrlx.policies as policies
from pfrlx.initializers import variancescaling_init


@hydra.main(config_path='../conf/mpo/default.yaml')
def main(conf: DictConfig):
    logging.basicConfig(level=conf.log.level)
    experiments.prepare_output_dir(args={}, basedir=os.getcwd(), exp_id='')
    print(f'Output files are saved in {os.getcwd()}')

    # Set a random seed
    utils.set_random_seed(conf.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(conf.n_envs) + conf.seed * conf.n_envs
    assert process_seeds.max() < 2 ** 32

    def make_dmc_mujoco_env(process_idx, test):
        if conf.dmc:
            return make_dmc_env(process_idx, test)
        else:
            return make_env(process_idx, test)

    def make_dmc_env(process_idx, test):
        import dmc2gym
        env = dmc2gym.make(
            domain_name=conf.domain, task_name=conf.task
        )
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = wrappers.CastObservationToFloat32(env)
        # Normalize action space to [-1, 1]^n
        env = wrappers.NormalizeActionSpace(env)
        if conf.log.monitor:
            env = gym.wrappers.Monitor(env, os.getcwd())
        if conf.log.render:
            env = wrappers.Render(env)
        return env

    def make_env(process_idx, test):
        env = gym.make(conf.env)
        # Unwrap TimeLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = wrappers.CastObservationToFloat32(env)
        # Normalize action space to [-1, 1]^n
        env = wrappers.NormalizeActionSpace(env)
        if conf.log.monitor:
            env = gym.wrappers.Monitor(env, os.getcwd())
        if conf.log.render:
            env = wrappers.Render(env)
        return env

    def make_batch_env(test):
        return envs.MultiprocessVectorEnv(
            [
                functools.partial(make_dmc_mujoco_env, idx, test)
                for idx, env in enumerate(range(conf.n_envs))
            ]
        )

    sample_env = make_dmc_mujoco_env(process_idx=0, test=False)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    explorer = policies.GaussianHeadWithDiagonalCovariance(init_scale=conf.actor.init_scale)
    policy = nn.Sequential(
        nn.Linear(obs_size, conf.actor.nn_size),
        nn.LayerNorm(conf.actor.nn_size),
        nn.Tanh(),
        nn.Linear(conf.actor.nn_size, conf.actor.nn_size),
        nn.ELU(),
        nn.Linear(conf.actor.nn_size, conf.actor.nn_size),
        nn.ELU(),
        nn.Linear(conf.actor.nn_size, action_size * 2),
        explorer,
    )
    variancescaling_init(policy[0].weight, scale=0.33, mode='fan_out', distribution='uniform')
    variancescaling_init(policy[3].weight, scale=0.33, mode='fan_out', distribution='uniform')
    variancescaling_init(policy[5].weight, scale=0.33, mode='fan_out', distribution='uniform')
    variancescaling_init(policy[7].weight, scale=1e-4)
    torch.nn.init.zeros_(policy[7].bias)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=conf.actor.lr)

    q_func = nn.Sequential(
        networks.ConcatObsAndAction(),
        nn.Linear(obs_size + action_size, conf.critic.nn_size),
        nn.LayerNorm(conf.critic.nn_size),
        nn.Tanh(),
        nn.Linear(conf.critic.nn_size, conf.critic.nn_size),
        nn.ELU(),
        nn.Linear(conf.critic.nn_size, int(conf.critic.nn_size / 2)),
        nn.ELU(),
        nn.Linear(int(conf.critic.nn_size / 2), 1),
    )
    variancescaling_init(q_func[1].weight, scale=0.33, mode='fan_out', distribution='uniform')
    variancescaling_init(q_func[4].weight, scale=0.33, mode='fan_out', distribution='uniform')
    variancescaling_init(q_func[6].weight, scale=0.33, mode='fan_out', distribution='uniform')
    variancescaling_init(q_func[8].weight, scale=0.33, mode='fan_out', distribution='uniform')
    q_func_optimizer = torch.optim.Adam(q_func.parameters(), lr=conf.critic.lr)

    rbuf = replay_buffers.ReplayBuffer(capacity=conf.replay_buffer.size)

    agent = MPO(
        conf,
        policy,
        q_func,
        policy_optimizer,
        q_func_optimizer,
        rbuf,
        action_space=action_space
    )

    if conf.pretrained.dir or conf.pretrained.load:
        if conf.pretrained.load:
            raise Exception("Pretrained models are currently unsupported.")
        # either dir or load must be false
        assert not (conf.pretrained.dir and conf.pretrained.load)
        if conf.pretrained.dir:
            agent.load(conf.pretrained.dir)
        else:
            agent.load(
                utils.download_model(
                    "MPO", conf.env, model_type="final"
                )[0]
            )

    if conf.demo:
        env = make_batch_env(True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=conf.eval.n_runs,
            max_episode_len=timestep_limit,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev {}".format(
                conf.eval.n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=os.getcwd(),
            steps=conf.steps,
            eval_n_steps=None,
            eval_n_episodes=conf.eval.n_runs,
            eval_interval=conf.eval.interval,
            log_interval=conf.log.interval,
            max_episode_len=timestep_limit,
            return_window_size=timestep_limit,
        )


if __name__ == "__main__":
    main()
