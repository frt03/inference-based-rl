from collections import deque
import logging
import os

import numpy as np


from pfrlx.experiments.tester import Tester
from pfrlx.experiments.tester import save_agent


def train_agent_offline_batch(
    agent,
    steps,
    outdir,
    checkpoint_freq=None,
    log_interval=None,
    max_episode_len=None,
    eval_interval=None,
    step_offset=0,
    tester=None,
    successful_score=None,
    step_hooks=(),
    logger=None,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        steps (int): Number of total time steps for training.
        outdir (str): Path to the directory to output things.
        checkpoint_freq (int): frequency at which agents are stored.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        eval_interval (int): Interval of evaluation.
        step_offset (int): Time step from which training starts.
        tester :
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrlx.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    try:
        while True:
            t += 1

            agent.offline_update(counter=t)

            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agent, t, outdir, logger, suffix="_checkpoint")

            for hook in step_hooks:
                hook(None, agent, t)

            if (log_interval is not None and t >= log_interval and t % log_interval == 0):
                logger.info(
                    "outdir:{} training roop:{}".format(  # NOQA
                        outdir,
                        t,
                    )
                )
                logger.info("statistics: {}".format(agent.get_statistics()))
            if tester:
                if tester.evaluate_if_necessary(t=t, episodes=0):
                    if (
                        successful_score is not None
                        and tester.max_score >= successful_score
                    ):
                        break

            if t >= steps:
                break

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        if tester:
            tester.env.close()
        raise
    else:
        # Save the final model
        save_agent(agent, t, outdir, logger, suffix="_finish")


def train_agent_offline_batch_with_evaluation(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    eval_max_episode_len=None,
    eval_env=None,
    log_interval=None,
    successful_score=None,
    step_hooks=(),
    save_best_so_far_agent=True,
    logger=None,
):
    """Train an agent while regularly evaluating it.

    Args:
        agent: Agent to train.
        steps (int): Number of total time steps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        checkpoint_freq (int): frequency with which to store networks
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        return_window_size (int): Number of training episodes used to estimate
            the average returns of the current agent.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If set to None, max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrlx.experiments.hooks.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)

    os.makedirs(outdir, exist_ok=True)

    if eval_max_episode_len is None:
        eval_max_episode_len = max_episode_len

    tester = Tester(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        save_best_so_far_agent=save_best_so_far_agent,
        logger=logger,
    )

    train_agent_offline_batch(
        agent,
        steps,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        eval_interval=eval_interval,
        tester=tester,
        successful_score=successful_score,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger,
    )
