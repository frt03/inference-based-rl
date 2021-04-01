import collections
from logging import getLogger
import random
import itertools

import numpy as np
import torch
from torch.nn import functional as F

import pfrlx.utils as utils
from pfrlx.algo import AttributeSavingMixin, BatchAlgo
from pfrlx.utils.batch_states import batch_states
from pfrlx.utils.mode_of_distribution import mode_of_distribution
from pfrlx.utils.mean_or_nan import mean_or_nan, var_or_nan, max_or_nan, min_or_nan


def _add_value_to_episodes(
    episodes, policy, vf, stabilizer, phi, batch_states, obs_normalizer, device,
):

    dataset = list(itertools.chain.from_iterable(episodes))

    # Compute v_pred and next_v_pred
    states = batch_states([b["state"] for b in dataset], device, phi)
    next_states = batch_states([b["next_state"] for b in dataset], device, phi)

    if obs_normalizer:
        states = obs_normalizer(states, update=False)
        next_states = obs_normalizer(next_states, update=False)

    with torch.no_grad(), utils.evaluating(policy), utils.evaluating(vf):
        distribs = policy(states)
        vs_pred = vf(states)
        next_vs_pred = vf(next_states)

        vs_pred = vs_pred.cpu().numpy().ravel()
        next_vs_pred = next_vs_pred.cpu().numpy().ravel()

    for transition, v_pred, next_v_pred in zip(
        dataset, vs_pred, next_vs_pred
    ):
        transition["v_pred"] = 1/(1 - stabilizer.gamma) * v_pred
        transition["next_v_pred"] = 1/(1 - stabilizer.gamma) * next_v_pred

def _make_dataset(
    episodes,
    policy,
    vf,
    stabilizer,
    phi,
    batch_states,
    obs_normalizer,
    device
):
    """Make a list of transitions with necessary information."""

    _add_value_to_episodes(
        episodes=episodes,
        policy=policy,
        vf=vf,
        stabilizer=stabilizer,
        phi=phi,
        batch_states=batch_states,
        obs_normalizer=obs_normalizer,
        device=device,
    )

    episodes = stabilizer.add_advantage_and_value_target_to_episodes(episodes)

    return list(itertools.chain.from_iterable(episodes))

def _yield_minibatches(dataset, minibatch_size, num_epochs):
    assert dataset
    buf = []
    n = 0

    # remove terminal transitions
    masks = [bool(transition["is_state_terminal"]) for transition in dataset]
    masked_idxs = [i for i, mask in enumerate(masks) if mask]
    for idx in reversed(masked_idxs):
        del dataset[idx]

    while n < minibatch_size * num_epochs:
        while len(buf) < minibatch_size:
            buf = random.sample(dataset, k=len(dataset)) + buf
        assert len(buf) >= minibatch_size
        yield buf[-minibatch_size:]
        n += minibatch_size
        buf = buf[:-minibatch_size]

class AWR(AttributeSavingMixin, BatchAlgo):
    """Advantage-weighted Regression(AWR).

    See https://arxiv.org/abs/1910.00177

    Args:
        policy (torch.nn.Module): Policy.
        vf (torch.nn.Module): Model to train
            state s  |->  v(s)
        policy_opt (torch.optim.Optimizer): Optimizer used to train the policy
        vf_opt (torch.optim.Optimizer): Optimizer used to train the vf
        stabilizer (object): Stabilizer
        gpu (int): GPU device id if not None nor negative
        phi (callable): Feature extractor function
        minibatch_size (int): Minibatch size
        value_update_steps (int): Training epochs of vf per each iteration
        policy_update_steps (int): Training epochs of policy per each iteration
        logger (Logger): Logger used
        batch_states (callable): method which makes a batch of observations.
            default is `pfrlx.utils.batch_states.batch_states`
        weight_clip (float): clipping threshold for weight exp(A)
        temperature (float): temperature beta
        standardize_advantages (bool): Use standardized advantages on updates
        act_deterministically (bool): If set to True, choose most probable
            actions in the act method instead of sampling from distributions.
        action_space (gym.space.Box): env.action_space
        policy_bound_loss_weight (float): weight of policy bound loss
    """
    saved_attributes = ("policy", "vf", "policy_optimizer", "vf_optimizer", "obs_normalizer")

    def __init__(
        self,
        conf,
        policy,
        vf,
        policy_opt,
        vf_opt,
        stabilizer,
        replay_buffer,
        obs_normalizer=None,
        phi=lambda x: x,
        logger=getLogger(__name__),
        batch_states=batch_states,
        action_space=None
    ):

        self.policy = policy
        self.vf = vf
        self.policy_optimizer = policy_opt
        self.vf_optimizer = vf_opt
        self.stabilizer = stabilizer
        self.obs_normalizer = obs_normalizer

        if conf.gpu is not None and conf.gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(conf.gpu))
            self.policy.to(self.device)
            self.vf.to(self.device)
            if self.obs_normalizer is not None:
                self.obs_normalizer.to(self.device)
        else:
            self.device = torch.device("cpu")

        self.replay_buffer = replay_buffer
        self.phi = phi
        self.gpu = conf.gpu
        self.logger = logger
        self.batch_states = batch_states
        self.minibatch_size = conf.agent.batch_size
        self.update_interval = conf.agent.update_interval
        self.value_update_steps = conf.critic.update_steps
        self.policy_update_steps = conf.actor.update_steps
        self.weight_clip = conf.actor.weight_clip
        self.temperature = conf.agent.temperature
        self.standardize_advantages = conf.agent.standardize_advantages
        self.act_deterministically = conf.act_deterministically
        self.action_space_high = torch.tensor(
            action_space.high, device=self.device
        )
        self.action_space_low = torch.tensor(
            action_space.low, device=self.device
        )

        self.policy_bound_loss_weight = conf.actor.bound_loss_weight
        self.samples_per_iter = conf.agent.samples_per_iter
        self.normalizer_samples = conf.normalizer.samples
        self.settings = conf.settings

        self.t = 0

        #  Batch versions of last_episode, last_state, and last_action
        self.batch_last_episode = None
        self.batch_last_obs = None
        self.batch_last_action = None

        # Statistics
        self.value_record = collections.deque(maxlen=1000)
        self.value_loss_record = collections.deque(maxlen=100)
        self.policy_loss_record = collections.deque(maxlen=100)
        if self.standardize_advantages:
            self.advantage_record = collections.deque(maxlen=1)
        self.adv_weight_record = collections.deque(maxlen=1000)
        self.n_policy_updates = 0
        self.n_vf_updates = 0

    def _update_obs_normalizer(self, batch):
        assert self.obs_normalizer
        states = self.batch_states([b["state"] for b in batch], self.device, self.phi)
        self.obs_normalizer.experience(states)

    def update_vf(self, episodes):
        episodes = _make_dataset(
            episodes=episodes,
            policy=self.policy,
            vf=self.vf,
            stabilizer=self.stabilizer,
            phi=self.phi,
            batch_states=self.batch_states,
            obs_normalizer=self.obs_normalizer,
            device=self.device,
        )

        if self.obs_normalizer and (self.t <= self.normalizer_samples):
            self._update_obs_normalizer(episodes)

        device = self.device

        assert "state" in episodes[0]
        assert "v_teacher" in episodes[0]

        num_epochs = int(np.ceil(self.value_update_steps * self.update_interval / self.samples_per_iter))

        for batch in _yield_minibatches(
            episodes, minibatch_size=self.minibatch_size, num_epochs=num_epochs
        ):
            states = self.batch_states(
                [b["state"] for b in batch], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)

            vs_pred = self.vf(states)

            # value normalization
            vs_teacher = torch.tensor(
                [(1 - self.stabilizer.gamma) * b["v_teacher"] for b in batch], dtype=torch.float, device=device,
            )

            # Same shape as vs_pred: (batch_size, 1)
            vs_teacher = vs_teacher[..., None]

            # compute loss
            loss = 0.5 * F.mse_loss(vs_pred, vs_teacher)
            self.value_loss_record.append(float(loss))
            self.vf_optimizer.zero_grad()
            loss.backward()
            self.vf_optimizer.step()

            self.n_vf_updates += 1

    def update_policy(self, episodes):
        episodes = _make_dataset(
            episodes=episodes,
            policy=self.policy,
            vf=self.vf,
            stabilizer=self.stabilizer,
            phi=self.phi,
            batch_states=self.batch_states,
            obs_normalizer=self.obs_normalizer,
            device=self.device,
        )

        device = self.device

        assert "state" in episodes[0]

        if self.standardize_advantages:
            all_advs = torch.tensor([b["adv"] for b in episodes], device=device)
            std_advs, mean_advs = torch.std_mean(all_advs, unbiased=False)
            self.advantage_record.append(float(mean_advs))

        num_epochs = int(np.ceil(self.policy_update_steps * self.update_interval / self.samples_per_iter))

        for batch in _yield_minibatches(
            episodes, minibatch_size=self.minibatch_size, num_epochs=num_epochs
        ):
            states = self.batch_states(
                [b["state"] for b in batch], self.device, self.phi
            )
            if self.obs_normalizer:
                states = self.obs_normalizer(states, update=False)
            actions = torch.tensor([b["action"] for b in batch], device=device)
            distribs = self.policy(states)

            advs = torch.tensor(
                [b["adv"] for b in batch], dtype=torch.float32, device=device
            )
            if self.standardize_advantages:
                advs = (advs - mean_advs) / (std_advs + 1e-8)

            weights = torch.exp(advs / self.temperature)
            self.adv_weight_record.extend(weights.detach().cpu().numpy())
            weights = torch.clamp(weights, max=self.weight_clip)
            loss_policy = -torch.mean(weights * distribs.log_prob(actions))
            self.policy_loss_record.append(float(loss_policy))

            if self.policy_bound_loss_weight > 0:
                val = mode_of_distribution(distribs)
                vio_min = torch.clamp(val - self.action_space_low, max=0)
                vio_max = torch.clamp(val - self.action_space_high, min=0)
                violation = vio_min.pow_(2).sum(axis=-1) + vio_max.pow_(2).sum(axis=-1)
                loss_bound = 0.5 * torch.mean(violation)
            else:
                loss_bound = 0

            loss = loss_policy + self.policy_bound_loss_weight * loss_bound
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

            self.n_policy_updates += 1


    def update_if_necessary(self, iteration, update_func):

        if len(self.replay_buffer) < self.minibatch_size:
            return False

        if iteration % self.update_interval != 0:
            return False

        episodes = self.replay_buffer.sample_episodes(self.replay_buffer.n_episodes, None)
        update_func(episodes)

        return True

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

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        with torch.no_grad(), utils.evaluating(self.policy), utils.evaluating(self.vf):
            action_distrib = self.policy(b_state)
            if self.act_deterministically:
                action = mode_of_distribution(action_distrib).cpu().numpy()
            else:
                action = action_distrib.sample().cpu().numpy()

        return action

    def _batch_act_train(self, batch_obs):
        assert self.training
        b_state = self.batch_states(batch_obs, self.device, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        # action_distrib will be recomputed when computing gradients
        with torch.no_grad(), utils.evaluating(self.policy), utils.evaluating(self.vf):
            action_distrib = self.policy(b_state)
            batch_value = self.vf(b_state)
            batch_action = action_distrib.sample().cpu().numpy()
            self.value_record.extend(batch_value.cpu().numpy())

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
                    is_state_terminal=batch_reset[i] or batch_done[i],
                    env_id=i,
                    nonterminal= 0.0 if batch_done[i] else 1.0,
                )
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None

        if not (self.settings == 'deployment'):
            # update value function
            self.update_if_necessary(self.t, self.update_vf)
            # update policy
            self.update_if_necessary(self.t, self.update_policy)

    def offline_update(self, counter):
        self.t += counter - self.t
        self.update_if_necessary(self.t, self.update_vf)
        self.update_if_necessary(self.t, self.update_policy)

    def get_statistics(self):
        if self.standardize_advantages:
            return [
                ("average_value", mean_or_nan(self.value_record)),
                ("average_advantage", mean_or_nan(self.advantage_record)),
                ("average_value_loss", mean_or_nan(self.value_loss_record)),
                ("average_policy_loss", mean_or_nan(self.policy_loss_record)),
                ("n_policy_updates", self.n_policy_updates),
                ("n_vf_updates", self.n_vf_updates),
                ("adv_weight_mean", mean_or_nan(self.adv_weight_record)),
                ("adv_weight_var", var_or_nan(self.adv_weight_record)),
                ("adv_weight_max", max_or_nan(self.adv_weight_record)),
                ("adv_weight_min", min_or_nan(self.adv_weight_record)),
            ]
        else:
            return [
                ("average_value", mean_or_nan(self.value_record)),
                ("average_value_loss", mean_or_nan(self.value_loss_record)),
                ("average_policy_loss", mean_or_nan(self.policy_loss_record)),
                ("n_policy_updates", self.n_policy_updates),
                ("n_vf_updates", self.n_vf_updates),
                ("adv_weight_mean", mean_or_nan(self.adv_weight_record)),
                ("adv_weight_var", var_or_nan(self.adv_weight_record)),
                ("adv_weight_max", max_or_nan(self.adv_weight_record)),
                ("adv_weight_min", min_or_nan(self.adv_weight_record)),
            ]
