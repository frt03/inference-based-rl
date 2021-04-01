import collections
import copy
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Independent

import pfrlx.utils as utils
from pfrlx.algo import AttributeSavingMixin, BatchAlgo
from pfrlx.utils import clip_l2_grad_norm_
from pfrlx.utils.batch_states import batch_states
from pfrlx.utils.copy_param import synchronize_parameters
from pfrlx.utils.mean_or_nan import mean_or_nan
from pfrlx.utils.mode_of_distribution import mode_of_distribution
from pfrlx.replay_buffer import batch_experiences, ReplayUpdater
from pfrlx.stabilizer.clipped_double_q import clipped_double_q


class AWAC(AttributeSavingMixin, BatchAlgo):
    """Advantage-Weighted Actor-Critic (AWAC).

    See https://arxiv.org/abs/2006.09359

    Args:
        policy (Policy): Policy.
        q_func1 (Module): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        q_func2 (Module): Second Q-function that takes state-action pairs as
            input and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func1_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        q_func2_optimizer (Optimizer): Optimizer setup with the second
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
        gamma (float): Discount factor
        gpu (int): GPU device id if not None nor negative.
        replay_start_size (int): if the replay buffer's size is less than
            replay_start_size, skip update
        minibatch_size (int): Minibatch size
        update_interval (int): Model update interval in step
        phi (callable): Feature extractor applied to observations
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrl.utils.batch_states.batch_states`
        burnin_action_func (callable or None): If not None, this callable
            object is used to select actions before the model is updated
            one or more times during training.
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
    """

    saved_attributes = (
        "policy",
        "q_func1",
        "q_func2",
        "target_q_func1",
        "target_q_func2",
        "policy_optimizer",
        "q_func1_optimizer",
        "q_func2_optimizer",
    )

    def __init__(
        self,
        conf,
        policy,
        q_func1,
        q_func2,
        policy_optimizer,
        q_func1_optimizer,
        q_func2_optimizer,
        replay_buffer,
        update_interval=1,
        phi=lambda x: x,
        max_grad_norm=None,
        logger=getLogger(__name__),
        batch_states=batch_states,
        burnin_action_func=None,
        act_deterministically=True,
    ):

        self.policy = policy
        self.q_func1 = q_func1
        self.q_func2 = q_func2

        if conf.gpu is not None and conf.gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(conf.gpu))
            self.policy.to(self.device)
            self.q_func1.to(self.device)
            self.q_func2.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.gamma = conf.agent.gamma
        self.gpu = conf.gpu
        self.phi = phi
        self.soft_update_tau = conf.critic.soft_update_tau
        self.lambd = conf.actor.lambd
        self.weight_clip = conf.actor.weight_clip
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func1_optimizer = q_func1_optimizer
        self.q_func2_optimizer = q_func2_optimizer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=conf.agent.batch_size,
            n_times_update=1,
            replay_start_size=conf.replay_buffer.start_size,
            update_interval=update_interval,
            episodic_update=False,
        )
        self.max_grad_norm = max_grad_norm
        self.batch_states = batch_states
        self.burnin_action_func = burnin_action_func
        self.act_deterministically = act_deterministically
        self._clipped_double_q = conf.critic.clipped_double_q

        self.t = 0

        # Target model
        self.target_q_func1 = copy.deepcopy(self.q_func1).eval().requires_grad_(False)
        self.target_q_func2 = copy.deepcopy(self.q_func2).eval().requires_grad_(False)

        # Statistics
        self.q1_record = collections.deque(maxlen=1000)
        self.q2_record = collections.deque(maxlen=1000)
        self.q_func1_loss_record = collections.deque(maxlen=100)
        self.q_func2_loss_record = collections.deque(maxlen=100)
        self.policy_loss_record = collections.deque(maxlen=100)
        self.adv_weight_record = collections.deque(maxlen=1000)
        self.policy_entropy_record = collections.deque(maxlen=1000)
        self.n_policy_updates = 0

    def _sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.q_func1,
            dst=self.target_q_func1,
            method="soft",
            tau=self.soft_update_tau,
        )
        if self._clipped_double_q:
            synchronize_parameters(
                src=self.q_func2,
                dst=self.target_q_func2,
                method="soft",
                tau=self.soft_update_tau,
            )

    def update(self, experiences, errors_out=None):
        """Update the model from experiences"""
        batch = batch_experiences(experiences, self.device, self.phi, self.gamma)

        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        # policy evaluation
        with torch.no_grad(), utils.evaluating(self.policy), utils.evaluating(
            self.target_q_func1
        ), utils.evaluating(self.target_q_func2):
            next_action_distrib = self.policy(batch_next_state)
            next_actions = next_action_distrib.sample()
            next_q1 = self.target_q_func1((batch_next_state, next_actions))
            if self._clipped_double_q:
                next_q2 = self.target_q_func2((batch_next_state, next_actions))

                target_q = clipped_double_q(
                    batch_rewards,
                    batch_discount,
                    batch_terminal,
                    next_q1,
                    next_q2
                )
            else:
                target_q = batch_rewards + batch_discount * (1.0 - batch_terminal) * torch.flatten(next_q1)

        predict_q1 = self.q_func1((batch_state, batch_actions))
        loss1 = 0.5 * F.mse_loss(target_q, torch.flatten(predict_q1))
        if self._clipped_double_q:
            predict_q2 = self.q_func2((batch_state, batch_actions))
            loss2 = 0.5 * F.mse_loss(target_q, torch.flatten(predict_q2))

        # policy improvement
        action_distrib = self.policy(batch_state)
        actions = action_distrib.sample()

        v1_pi = self.q_func1((batch_state, actions))
        if self._clipped_double_q:
            v2_pi = self.q_func2((batch_state, actions))
            v_pi = torch.min(v1_pi, v2_pi)
            q_adv = torch.min(predict_q1, predict_q2)
        else:
            v_pi = v1_pi
            q_adv = predict_q1

        weights = F.softmax((q_adv - v_pi) / self.lambd, dim=0)[:, 0].detach() * len(q_adv)
        clipped_weights = torch.clamp(weights, max=self.weight_clip)
        policy_loss =  -torch.mean(clipped_weights * action_distrib.log_prob(batch_actions))

        # Update stats
        self.q1_record.extend(predict_q1.detach().cpu().numpy())
        self.q_func1_loss_record.append(float(loss1))
        self.policy_loss_record.append(float(policy_loss))
        self.adv_weight_record.extend(weights.cpu().numpy())

        # Update params
        self.q_func1_optimizer.zero_grad()
        loss1.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func1.parameters(), self.max_grad_norm)
        self.q_func1_optimizer.step()

        if self._clipped_double_q:
            self.q2_record.extend(predict_q2.detach().cpu().numpy())
            self.q_func2_loss_record.append(float(loss2))

            self.q_func2_optimizer.zero_grad()
            loss2.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.q_func2.parameters(), self.max_grad_norm)
            self.q_func2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        self.n_policy_updates += 1

        if isinstance(action_distrib, Normal) or isinstance(action_distrib, Independent):
            self.policy_entropy_record.append(float(action_distrib.entropy().mean()))
        else:
            sampled_action = action_distrib.sample()
            entropy_estimate = - action_distrib.log_prob(sampled_action).mean()
            self.policy_entropy_record.append(float(entropy_estimate))

        # Update target network
        self._sync_target_network()

    def batch_select_greedy_action(self, batch_obs, deterministic=False):
        with torch.no_grad(), utils.evaluating(self.policy):
            batch_xs = self.batch_states(batch_obs, self.device, self.phi)
            policy_out = self.policy(batch_xs)
            if deterministic:
                batch_action = mode_of_distribution(policy_out).cpu().numpy()
            else:
                batch_action = policy_out.sample().cpu().numpy()
        return batch_action

    def batch_act(self, batch_obs):
        if self.training:
            return self._batch_act_train(batch_obs)
        else:
            return self._batch_act_eval(batch_obs)

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        if self.training:
            self._batch_observe_train(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_act_eval(self, batch_obs):
        assert not self.training
        return self.batch_select_greedy_action(
            batch_obs, deterministic=self.act_deterministically
        )

    def _batch_act_train(self, batch_obs):
        assert self.training
        if self.burnin_action_func is not None and self.n_policy_updates == 0:
            batch_action = [self.burnin_action_func() for _ in range(len(batch_obs))]
        else:
            batch_action = self.batch_select_greedy_action(batch_obs)
        self.batch_last_obs = list(batch_obs)
        self.batch_last_action = list(batch_action)

        return batch_action

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        assert self.training
        for i in range(len(batch_obs)):
            self.t += 1
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # Add a transition to the replay buffer
                self.replay_buffer.append(
                    state=self.batch_last_obs[i],
                    action=self.batch_last_action[i],
                    reward=batch_reward[i],
                    next_state=batch_obs[i],
                    next_action=None,
                    is_state_terminal=batch_done[i],
                    env_id=i,
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.replay_buffer.stop_current_episode(env_id=i)
            self.replay_updater.update_if_necessary(self.t)

    def get_statistics(self):
        return [
            ("average_q1", mean_or_nan(self.q1_record)),
            ("average_q2", mean_or_nan(self.q2_record)),
            ("average_q_func1_loss", mean_or_nan(self.q_func1_loss_record)),
            ("average_q_func2_loss", mean_or_nan(self.q_func2_loss_record)),
            ("n_updates", self.n_policy_updates),
            ("average_policy_loss", mean_or_nan(self.policy_loss_record)),
            ("adv_weight_mean", mean_or_nan(self.adv_weight_record)),
            ("average_policy_entropy_mean", mean_or_nan(self.policy_entropy_record)),
        ]
