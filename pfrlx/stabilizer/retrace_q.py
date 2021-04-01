import numpy as np


class RetraceQ(object):
    def __init__(self, gamma=0.99, lambd=1.0, n=10):
        '''This calculates Retrace targets for a state-value function.

        This is an approximated version of Retrace(lambda). The original
        Retrace(lambda) computes Retrace targets by an infinite-sum of
        TD errors (decayed by gamma lambda clipped_IS_ratio). However,
        this function returns an n-step sum of TD errors (decayed by
        gamma lambda clipped_IS_ratio).

        Args:
            gamma (float, optional): Discount factor. Defaults to 0.99.
            lambd (float, optional): Lambda. Defaults to 1.0
            n (int, optional): n used in the computation of Retrace targets
                as explained above.
        '''
        self.gamma = gamma
        self.lambd = lambd
        self.gamma_lambd = gamma * lambd
        self.n = n  # IS horizon

    def _add_target_retrace_q_to_episode(self, episode):
        """Compute retrace Q for an episode."""
        horizon = len(episode)
        for time in range(horizon):
            cum_td = 0.0
            ck = 0.0
            for t in reversed(episode[time:min(time + self.n, horizon)]):
                mask = t["nonterminal"]
                q_est = t["reward"] + self.gamma * mask * t["next_q_pred"]
                # Note that ck of the next time step, not this time step,
                # must be used here.
                t["retrace_q"] = q_est + self.gamma_lambd * mask * ck * cum_td
                cum_td = t["retrace_q"] - t["q_pred"]

                # The following ck is a stable version of
                #     min(np.exp(t["current_log_prob"] - t["log_prob"]), 1).
                ck = np.exp(
                    min(t["current_log_prob"] - t["log_prob"], 0)
                )

    def add_target_retrace_q_to_episodes(self, episodes):
        """Add target retrace Q values to a list of episodes."""
        for episode in episodes:
            self._add_target_retrace_q_to_episode(episode)
        return episodes
