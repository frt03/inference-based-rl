import numpy as np

def load_d4rl_dataset(replay_buffer, d4rl_env, is_goal=False):
    import d4rl
    dataset = d4rl.qlearning_dataset(d4rl_env)
    obss = dataset['observations']
    acts = dataset['actions']
    next_obss = dataset['next_observations']
    rs = dataset['rewards']
    terms = dataset['terminals']
    if 'timeouts' not in dataset:
        timeouts = np.array([False] * len(obss))
    else:
        timeouts = dataset['timeouts']
    if is_goal:
        dataset_raw = env.get_dataset()
        if 'infos/goal' not in dataset_raw:
            raise ValueError('Only Goal Conditional environments can be used')
        goals = dataset_raw['infos/goal']
        for obs, act, next_obs, r, term, to, goal in zip(obss, acts, next_obss, rs, terms, timeouts, goals):
            replay_buffer.append(
                state=obs,
                action=act,
                reward=r,
                next_state=next_obs,
                next_action=None,
                is_state_terminal=to or term,
                env_id=0,
                desired_goal=goal,
                nonterminal= 0.0 if term else 1.0,
            )
    else:
        for obs, act, next_obs, r, term, to in zip(obss, acts, next_obss, rs, terms, timeouts):
            replay_buffer.append(
                state=obs,
                action=act,
                reward=r,
                next_state=next_obs,
                next_action=None,
                is_state_terminal=to or term,
                env_id=0,
                nonterminal= 0.0 if term else 1.0,
            )
    print('Dataset loding is finished.')
    return replay_buffer

