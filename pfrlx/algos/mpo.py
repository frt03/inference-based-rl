import collections
import copy
from logging import getLogger

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal, Independent

import pfrlx.utils as utils
from pfrlx.algo import AttributeSavingMixin
from pfrlx.algo import BatchAlgo
from pfrlx.utils.batch_states import batch_states
from pfrlx.utils.copy_param import synchronize_parameters
from pfrlx.utils.mode_of_distribution import mode_of_distribution
from pfrlx.replay_buffer import batch_experiences
from pfrlx.replay_buffer import ReplayUpdater
from pfrlx.utils import clip_l2_grad_norm_
from pfrlx.utils.mean_or_nan import mean_or_nan, var_or_nan, max_or_nan, min_or_nan


class TemperatureHolder(nn.Module):
    """Module that holds a temperature as a learnable value.

    Args:
        initial_temperature (float): Initial value(temperature).
    """

    def __init__(self, initial_log_temperature=1.0, dim=None):
        super().__init__()
        if dim is None:
            self.log_temperature = nn.Parameter(
                torch.tensor(initial_log_temperature, dtype=torch.float32)
            )
        else:
            self.log_temperature = nn.Parameter(
                torch.tensor(
                    [initial_log_temperature for _ in range(dim)],
                    dtype=torch.float32
                )
            )

    def forward(self):
        """Return a temperature as a torch.Tensor."""
        # clipping
        log_temperature = torch.clamp(self.log_temperature, min=-18.0)
        # Transform dual variables from log-space.
        # Note: using softplus instead of exponential for numerical stability.
        return F.softplus(log_temperature) + 1e-8


class MPO(AttributeSavingMixin, BatchAlgo):
    """Maximum a Posteriori Policy Optimisation (MPO).

    See https://arxiv.org/abs/1806.06920

    Args:
        conf: configuation of Hydra
        policy (Policy): Policy.
        q_func (Module): First Q-function that takes state-action pairs as input
            and outputs predicted Q-values.
        policy_optimizer (Optimizer): Optimizer setup with the policy
        q_func_optimizer (Optimizer): Optimizer setup with the first
            Q-function.
        replay_buffer (ReplayBuffer): Replay buffer
    """

    saved_attributes = (
        "policy",
        "q_func",
        "target_q_func",
        "policy_optimizer",
        "q_func_optimizer",
        "temperature_holder",
        "eta_mu_holder",
        "eta_var_holder",
        "temperature_optimizer",
        "eta_mu_optimizer",
        "eta_var_optimizer",
    )

    def __init__(
        self,
        conf,
        policy,
        q_func,
        policy_optimizer,
        q_func_optimizer,
        replay_buffer,
        logger=getLogger(__name__),
        batch_states=batch_states,
        phi=lambda x: x,
        action_space=None,
        max_grad_norm=40.0,  # from MPO in acme
        action_bound=True,  # from MPO in acme
    ):
        self.policy = policy
        self.q_func = q_func

        self.replay_buffer = replay_buffer
        self.replay_updater = ReplayUpdater(
            replay_buffer=replay_buffer,
            update_func=self.update,
            batchsize=conf.agent.batch_size,
            n_times_update=conf.agent.update_steps,
            replay_start_size=conf.replay_buffer.start_size,
            update_interval=conf.agent.update_interval,
            episodic_update=False,
        )
        self.gpu = conf.gpu
        self.phi = phi

        if self.gpu is not None and self.gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(conf.gpu))
            self.policy.to(self.device)
            self.q_func.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.num_action_samples = conf.agent.action_samples

        self.dual_constraint = conf.actor.dual_constraint
        self.mean_constraint = conf.actor.mean_constraint
        self.var_constraint = conf.actor.var_constraint
        self.action_penalty_constraint = conf.actor.action_penalty_constraint

        self.target_critic_update_intervals = conf.critic.target_critic_update_intervals
        self.target_actor_update_intervals = conf.actor.target_actor_update_intervals
        self.logger = logger
        self.policy_optimizer = policy_optimizer
        self.q_func_optimizer = q_func_optimizer
        self.max_grad_norm = max_grad_norm
        self.batch_states = batch_states
        self.act_deterministically = conf.act_deterministically
        assert action_space is not None
        self.action_space_high = torch.tensor(action_space.high, device=self.device)
        self.action_space_low = torch.tensor(action_space.low, device=self.device)
        self.action_dim = action_space.shape[0]
        self.action_bound = action_bound
        self.gamma = conf.agent.gamma
        self.batch_size = conf.agent.batch_size

        # lagrange multipliers
        self.temperature_holder = TemperatureHolder(
            initial_log_temperature=conf.agent.init_log_temperature,
        )
        self.action_penalty_holder = TemperatureHolder(
            initial_log_temperature=conf.agent.init_log_temperature,
        )
        self.eta_mu_holder = TemperatureHolder(
            initial_log_temperature=conf.actor.init_log_eta_mu,
            dim=self.action_dim
        )
        self.eta_var_holder = TemperatureHolder(
            initial_log_temperature=conf.actor.init_log_eta_var,
            dim=self.action_dim
        )
        self.temperature_optimizer = torch.optim.Adam(
            self.temperature_holder.parameters(), lr=conf.agent.temperature_lr,
        )
        self.eta_mu_optimizer = torch.optim.Adam(
            self.eta_mu_holder.parameters(), lr=conf.actor.eta_lr,
        )
        self.eta_var_optimizer = torch.optim.Adam(
            self.eta_var_holder.parameters(), lr=conf.actor.eta_lr,
        )
        self.action_penalty_optimizer = torch.optim.Adam(
            self.action_penalty_holder.parameters(), lr=conf.agent.temperature_lr,
        )

        if self.gpu is not None and self.gpu >= 0:
            self.temperature_holder.to(self.device)
            self.action_penalty_holder.to(self.device)
            self.eta_mu_holder.to(self.device)
            self.eta_var_holder.to(self.device)

        self.t = 0

        # Target model
        self.target_q_func = copy.deepcopy(self.q_func).eval().requires_grad_(False)
        self.target_policy = copy.deepcopy(self.policy).eval().requires_grad_(False)

        # Statistics
        self.q_record = collections.deque(maxlen=125)
        self.q_func_loss_record = collections.deque(maxlen=125)
        self.policy_loss_record = collections.deque(maxlen=125)
        self.normalized_weight_record = collections.deque(maxlen=125)
        self.policy_stddev_record = collections.deque(maxlen=1)
        self.policy_mean_record = collections.deque(maxlen=1)
        self.c_mu_record = collections.deque(maxlen=1)
        self.c_sigma_record = collections.deque(maxlen=1)
        self.n_policy_updates = 0
        self.n_q_updates = 0

    @property
    def temperature(self):
        with torch.no_grad():
            return float(self.temperature_holder())

    @property
    def eta_mu(self):
        with torch.no_grad():
            return self.eta_mu_holder()

    @property
    def eta_var(self):
        with torch.no_grad():
            return self.eta_var_holder()

    @property
    def action_penalty(self):
        with torch.no_grad():
            return float(self.action_penalty_holder())

    def _sync_target_network(self):
        """Synchronize target network with current network."""
        synchronize_parameters(
            src=self.q_func,
            dst=self.target_q_func,
            method="hard",
        )

    def _sync_target_policy(self):
        """Synchronize target policy with current policy."""
        synchronize_parameters(
            src=self.policy,
            dst=self.target_policy,
            method="hard",
        )

    def update(self, experiences):
        batch = batch_experiences(experiences, self.device, self.phi, self.gamma)
        self._update(batch)
        if self.n_policy_updates % self.target_actor_update_intervals == 0:
            self._sync_target_policy()
        if self.n_q_updates % self.target_critic_update_intervals == 0:
            self._sync_target_network()

    def _update(self, batch):
        batch_next_state = batch["next_state"]
        batch_rewards = batch["reward"]
        batch_terminal = batch["is_state_terminal"]
        batch_state = batch["state"]
        batch_actions = batch["action"]
        batch_discount = batch["discount"]

        state_dim = batch_state.shape[-1]

        # [num_action_samples * batch, state_dim]
        multi_next_states = batch_next_state[None].expand(
            self.num_action_samples, self.batch_size, state_dim
        ).reshape(self.num_action_samples * self.batch_size, state_dim)

        with torch.no_grad(), utils.evaluating(self.target_policy), utils.evaluating(self.target_q_func):
            multi_old_distribs = self.target_policy(multi_next_states)
            # [num_action_samples * batch, action_dim]
            multi_next_actions = multi_old_distribs.sample().reshape(self.num_action_samples * self.batch_size, self.action_dim)
            # [num_action_samples * batch, 1]
            next_qs_pred = self.target_q_func((multi_next_states, multi_next_actions))
            # [num_action_samples, batch, 1]
            next_qs_pred = torch.reshape(next_qs_pred, (self.num_action_samples, self.batch_size, 1))
            qs_target = batch_rewards + batch_discount * (1.0 - batch_terminal) * torch.flatten(torch.mean(next_qs_pred, dim=0))

        # policy evaluation
        qs_pred = torch.flatten(self.q_func((batch_state, batch_actions)))
        assert not qs_target.requires_grad
        assert qs_pred.requires_grad
        loss_q = 0.5 * F.mse_loss(qs_pred, qs_target)

        self.q_record.extend(qs_pred.detach().cpu().numpy())
        self.q_func_loss_record.append(float(loss_q))

        self.q_func_optimizer.zero_grad()
        loss_q.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.q_func.parameters(), self.max_grad_norm)
        self.q_func_optimizer.step()
        self.n_q_updates += 1

        # policy improvement
        # E-Step
        # [batch, 1]
        q_logsumexp = torch.logsumexp(next_qs_pred / self.temperature_holder(), dim=0)
        log_num_actions = torch.log(torch.tensor(self.num_action_samples, dtype=torch.float32))

        loss_e_step = self.temperature_holder() * (
            self.dual_constraint + torch.mean(q_logsumexp) - log_num_actions
        )

        # M-Step
        # Compute the normalized importance weights used to compute expectations with
        # respect to the non-parametric policy.
        # [num_action_samples, batch, 1]
        normalized_weights = torch.softmax(next_qs_pred / self.temperature, dim=0).detach()
        self.normalized_weight_record.extend(normalized_weights.cpu().numpy())

        # action bound loss
        if self.action_bound:
            vio_min = torch.clamp(multi_next_actions - self.action_space_low, max=0)
            vio_max = torch.clamp(multi_next_actions - self.action_space_high, min=0)
            diff_out_of_bound = vio_min + vio_max
            # [num_action_samples * batch, 1]
            cost_out_of_bound = -torch.norm(diff_out_of_bound, dim=-1)
            # [num_action_samples, batch, 1]
            cost_out_of_bound = torch.reshape(cost_out_of_bound, (self.num_action_samples, self.batch_size, 1))
            # [batch, 1]
            cost_logsumexp = torch.logsumexp(cost_out_of_bound / self.action_penalty_holder(), dim=0)

            loss_penalty = self.action_penalty_holder() * (
                self.action_penalty_constraint + torch.mean(cost_logsumexp) - log_num_actions
            )
            self.action_penalty_optimizer.zero_grad()
            loss_penalty.backward()
            if self.max_grad_norm is not None:
                clip_l2_grad_norm_(self.action_penalty_holder.parameters(), self.max_grad_norm)
            self.action_penalty_optimizer.step()

            normalized_weights += torch.softmax(cost_out_of_bound / self.action_penalty, dim=0).detach()

        # [num_action_samples * batch, 1]
        distribs = self.policy(multi_next_states)
        # acme MPO
        online_mean = distribs.base_dist.loc
        online_scale = distribs.base_dist.scale
        target_mean = multi_old_distribs.base_dist.loc
        target_scale = multi_old_distribs.base_dist.scale
        fixed_stddev_distribution = Independent(Normal(loc=online_mean, scale=target_scale), 1)
        fixed_mean_distribution = Independent(Normal(loc=target_mean, scale=online_scale), 1)
        log_prob_mean = torch.reshape(
            fixed_stddev_distribution.log_prob(multi_next_actions),
            (self.num_action_samples, self.batch_size, 1)
        )
        assert not normalized_weights.requires_grad
        assert log_prob_mean.requires_grad
        loss_policy_mean = torch.mean(-torch.sum(normalized_weights * log_prob_mean, dim=0))
        log_prob_stddev = torch.reshape(
            fixed_mean_distribution.log_prob(multi_next_actions),
            (self.num_action_samples, self.batch_size, 1)
        )
        assert log_prob_stddev.requires_grad
        loss_policy_stddev = torch.mean(-torch.sum(normalized_weights * log_prob_stddev, dim=0))

        # [batch, action_dim]
        kl_mean = torch.distributions.kl_divergence(multi_old_distribs.base_dist, fixed_stddev_distribution.base_dist)
        kl_stddev = torch.distributions.kl_divergence(multi_old_distribs.base_dist, fixed_mean_distribution.base_dist)
        loss_kl_mean = torch.sum(self.eta_mu * torch.mean(kl_mean, dim=0))
        loss_kl_stddev = torch.sum(self.eta_var * torch.mean(kl_stddev, dim=0))
        self.c_mu_record.append(float(torch.sum(torch.mean(kl_mean, dim=0))))
        self.c_sigma_record.append(float(torch.sum(torch.mean(kl_stddev, dim=0))))
        self.policy_mean_record.append(float(torch.mean(distribs.base_dist.loc)))
        self.policy_stddev_record.append(float(torch.mean(distribs.base_dist.scale)))

        loss_m_step = loss_policy_mean + loss_policy_stddev + loss_kl_mean + loss_kl_stddev
        self.policy_loss_record.append(float(loss_m_step))

        # compute loss
        self.policy_optimizer.zero_grad()
        loss_m_step.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        self.n_policy_updates += 1

        self.temperature_optimizer.zero_grad()
        loss_e_step.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.temperature_holder.parameters(), self.max_grad_norm)
        self.temperature_optimizer.step()

        # update lagrange multipliers eta_mu and eta_var
        loss_eta_mu = torch.sum(self.eta_mu_holder() *(self.mean_constraint - torch.mean(kl_mean, dim=0).detach()))
        self.eta_mu_optimizer.zero_grad()
        loss_eta_mu.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.eta_mu_holder.parameters(), self.max_grad_norm)
        self.eta_mu_optimizer.step()

        loss_eta_var = torch.sum(self.eta_var_holder() *(self.var_constraint - torch.mean(kl_stddev, dim=0).detach()))
        self.eta_var_optimizer.zero_grad()
        loss_eta_var.backward()
        if self.max_grad_norm is not None:
            clip_l2_grad_norm_(self.eta_var_holder.parameters(), self.max_grad_norm)
        self.eta_var_optimizer.step()

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
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        with torch.no_grad(), utils.evaluating(self.policy):
            action_distrib = self.policy(b_state)
            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()

        return action

    def _batch_act_train(self, batch_obs):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        with torch.no_grad(), utils.evaluating(self.policy):
            batch_action = self.policy(b_state).sample().cpu().numpy()

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
            ("average_q", mean_or_nan(self.q_record)),
            ("average_q_func_loss", mean_or_nan(self.q_func_loss_record)),
            ("average_policy_loss", mean_or_nan(self.policy_loss_record)),
            ("c_mu", mean_or_nan(self.c_mu_record)),
            ("c_sigma", mean_or_nan(self.c_sigma_record)),
            ("policy_stddev", mean_or_nan(self.policy_stddev_record)),
            ("policy_mean", mean_or_nan(self.policy_mean_record)),
            ("n_policy_updates", self.n_policy_updates),
            ("n_q_updates", self.n_q_updates),
            ("temperature", self.temperature),
            ("eta_mu", float(torch.mean(self.eta_mu).cpu())),
            ("eta_var", float(torch.mean(self.eta_var).cpu())),
            ("average_normalized_weight_mean", mean_or_nan(self.normalized_weight_record)),
            ("average_normalized_weight_var", var_or_nan(self.normalized_weight_record)),
            ("average_normalized_weight_max", max_or_nan(self.normalized_weight_record)),
            ("average_normalized_weight_min", min_or_nan(self.normalized_weight_record)),
        ]
