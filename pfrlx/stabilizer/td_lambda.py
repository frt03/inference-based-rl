class TD_Lambda(object):
    def __init__(self, gamma=0.99, lambd=0.95):
        '''This calculates Lambda return for a state-value function.

        Args:
            gamma (float, optional): Discount factor. Defaults to 0.99.
            lambd (float, optional): Lambda as in TD Lambda. Defaults to 0.95.
        '''
        self.gamma = gamma
        self.lambd = lambd
        self.gamma_lambd = gamma * lambd

    def _add_advantage_and_value_target_to_episode(self, episode):
        """Compute Lambda return for an episode."""
        cum_td = 0.0
        for t in reversed(episode):
            mask = t["nonterminal"]
            q_est = t["reward"] + self.gamma * mask * t["next_v_pred"]
            t["v_teacher"] = q_est + self.gamma_lambd * mask * cum_td
            t["adv"] = t["v_teacher"] - t["v_pred"]
            cum_td = t["v_teacher"] - t["v_pred"]

    def add_advantage_and_value_target_to_episodes(self, episodes):
        """Add advantage and value target values to a list of episodes."""
        for episode in episodes:
            self._add_advantage_and_value_target_to_episode(episode)

        return episodes
