import torch

def clipped_double_q(batch_rewards, batch_discount, batch_terminal, next_q1, next_q2, entropy_term=0):
    next_q = torch.min(next_q1, next_q2)
    target_q = batch_rewards + batch_discount * (1.0 - batch_terminal) * torch.flatten(next_q - entropy_term)

    return target_q