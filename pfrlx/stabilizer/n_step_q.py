class NstepQ(object):
    def __init__(self, gamma=0.99, n=10):
        '''This calculates N-step Q value

        Args:
            gamma (float, optional): Discount factor. Defaults to 0.99.
        '''
        self.gamma = gamma
        self.n = n

    def _add_target_n_step_q_to_episode(self, episode):
        """Compute n_step Q for an episode."""
        horizon = len(episode)
        for t in range(horizon):
            end_idx = min(t + self.n, horizon)
            discount = self.gamma ** (end_idx - t)
            mask = episode[end_idx - 1]["nonterminal"]
            episode[t]["n_step_q"] = discount * mask * episode[end_idx - 1]["next_q_pred"]
            for j in range(t, end_idx):
                if j == t:
                    mask = 1.0
                discount = self.gamma ** (j - t)
                episode[t]["n_step_q"] += discount * mask * episode[j]["reward"]
                mask = episode[j]["nonterminal"]

    def add_target_n_step_q_to_episodes(self, episodes):
        """Add target n-step Q values to a list of episodes."""
        for episode in episodes:
            self._add_target_n_step_q_to_episode(episode)
        return episodes
