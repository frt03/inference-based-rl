"""
A training script of Advantage-Weighted Regression on OpenAI Gym Mujoco environments.

This script follows the settings of https://arxiv.org/abs/1910.00177 as much
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

from pfrlx.algos import AWR
from pfrlx.stabilizer import TD_Lambda
import pfrlx.experiments as experiments
import pfrlx.utils as utils
import pfrlx.replay_buffers as replay_buffers
import pfrlx.envs as envs
import pfrlx.wrappers as wrappers
import pfrlx.networks as networks
import pfrlx.policies as policies


@hydra.main(config_path='../conf/awr/default.yaml')
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

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = networks.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5
    )

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    if LooseVersion(torch.__version__) < LooseVersion("1.5.0"):
        raise Exception("This script requires a PyTorch version >= 1.5.0")

    explorer = policies.GaussianHeadWithFixedCovariance(
        scale=conf.actor.action_std
    )

    policy = nn.Sequential(
        nn.Linear(obs_size, conf.actor.nn_size),
        nn.ReLU(),
        nn.Linear(conf.actor.nn_size, int(conf.actor.nn_size / 2)),
        nn.ReLU(),
        nn.Linear(int(conf.actor.nn_size / 2), action_size),
        explorer,
    )
    torch.nn.init.xavier_uniform_(policy[0].weight)
    torch.nn.init.xavier_uniform_(policy[2].weight)
    torch.nn.init.uniform_(
        policy[4].weight,
        a=-conf.actor.output_scale,
        b=conf.actor.output_scale
    )
    torch.nn.init.zeros_(policy[4].bias)

    policy_optimizer = torch.optim.SGD(
        policy.parameters(), lr=conf.actor.lr, momentum=conf.actor.momentum
    )
    '''
    policy_optimizer = torch.optim.RMSprop(
        policy.parameters(), lr=conf.actor.lr, momentum=conf.actor.momentum
    )
    '''

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, conf.critic.nn_size),
        nn.ReLU(),
        nn.Linear(conf.critic.nn_size, int(conf.critic.nn_size / 2)),
        nn.ReLU(),
        nn.Linear(int(conf.critic.nn_size / 2), 1),
    )

    torch.nn.init.xavier_uniform_(vf[0].weight)
    torch.nn.init.xavier_uniform_(vf[2].weight)
    torch.nn.init.xavier_uniform_(vf[4].weight)

    vf_optimizer = torch.optim.SGD(
        vf.parameters(), lr=conf.critic.lr, momentum=conf.critic.momentum
    )
    '''
    vf_optimizer = torch.optim.RMSprop(
        vf.parameters(), lr=conf.critic.lr, momentum=conf.critic.momentum
    )
    '''

    # select stabilizer
    stabilizer = TD_Lambda(gamma=conf.agent.gamma, lambd=conf.agent.lambd)

    rbuf = replay_buffers.EpisodicReplayBuffer(capacity=50000)

    agent = AWR(
        conf,
        policy,
        vf,
        policy_optimizer,
        vf_optimizer,
        stabilizer,
        rbuf,
        obs_normalizer=obs_normalizer,
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
                    "AWR", conf.env, model_type="final"
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
