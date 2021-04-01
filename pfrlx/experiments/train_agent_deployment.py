from collections import deque
import logging
import os

import numpy as np


from pfrlx.experiments.tester import Tester
from pfrlx.experiments.tester import save_agent


def train_agent_deployment_batch(
    agent,
    env,
    num_batch,
    offline_update,
    num_deployment,
    use_dataset,
    outdir,
    checkpoint_freq=None,
    log_interval=None,
    max_episode_len=None,
    eval_interval=None,
    update_offset=0,
    tester=None,
    successful_score=None,
    step_hooks=(),
    logger=None,
):
    """Train an agent in a batch environment.

    Args:
        agent: Agent to train.
        num_batch (int): Batchsize collecting per deployment.
        offline_update (int): Number of offline update per deployment.
        num_deployment (int): Number of deployment.
        outdir (str): Path to the directory to output things.
        checkpoint_freq (int): frequency at which agents are stored.
        log_interval (int): Interval of logging.
        max_episode_len (int): Maximum episode length.
        eval_interval (int): Interval of evaluation.
        update_offset (int): Time step from which training starts.
        tester :
        successful_score (float): Finish training if the mean score is greater
            or equal to thisvalue if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrlx.experiments.hooks.
        logger (logging.Logger): Logger used in this function.
    """

    logger = logger or logging.getLogger(__name__)
    num_envs = env.num_envs
    episode_len = np.zeros(num_envs, dtype="i")

    t = update_offset
    if hasattr(agent, "t"):
        agent.t = update_offset

    try:
        for n in range(num_deployment):
            if (n != 0) or (n == 0 and not use_dataset):
                # o_0, r_0
                obss = env.reset()
                sample_counter = 0
                while True:
                    # a_t
                    actions = agent.batch_act(obss)
                    # o_{t+1}, r_{t+1}
                    obss, rs, dones, infos = env.step(actions)
                    episode_len += 1

                    # Compute mask for done and reset
                    if max_episode_len is None:
                        resets = np.zeros(num_envs, dtype=bool)
                    else:
                        resets = episode_len == max_episode_len
                    resets = np.logical_or(
                        resets, [info.get("needs_reset", False) for info in infos]
                    )
                    # Agent observes the consequences
                    agent.batch_observe(obss, rs, dones, resets)

                    # Make mask. 0 if done/reset, 1 if pass
                    end = np.logical_or(resets, dones)
                    not_end = np.logical_not(end)

                    sample_counter += num_envs

                    if sample_counter >= num_batch:
                        logger.info("Collected samples: {}".format(sample_counter))
                        break

                    # reset
                    episode_len[end] = 0
                    obss = env.reset(not_end)

            while True :
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

                if t >= offline_update * (n+1):
                    logger.info("Finish Uptdate: {}/{}".format(n+1, num_deployment))
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


def train_agent_deployment_batch_with_evaluation(
    agent,
    env,
    num_batch,
    offline_update,
    num_deployment,
    use_dataset,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    update_offset=0,
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
        num_batch (int): Batchsize collecting per deployment.
        offline_update (int): Number of offline update per deployment.
        num_deployment (int): Number of deployment.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_runs (int): Number of runs for each time of evaluation.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output things.
        log_interval (int): Interval of logging.
        checkpoint_freq (int): frequency with which to store networks
        max_episode_len (int): Maximum episode length.
        update_offset (int): Time step from which training starts.
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
        step_offset=update_offset,  # TODO: check?
        save_best_so_far_agent=save_best_so_far_agent,
        logger=logger,
    )

    train_agent_deployment_batch(
        agent,
        env,
        num_batch,
        offline_update,
        num_deployment,
        use_dataset,
        outdir,
        checkpoint_freq=checkpoint_freq,
        max_episode_len=max_episode_len,
        update_offset=update_offset,
        eval_interval=eval_interval,
        tester=tester,
        successful_score=successful_score,
        log_interval=log_interval,
        step_hooks=step_hooks,
        logger=logger,
    )
