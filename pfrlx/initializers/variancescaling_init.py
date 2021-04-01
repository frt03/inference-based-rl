import math
import torch


def variancescaling_init(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    dimensions = tensor.dim()
    assert dimensions == 2
    # only for Linear
    _fan_in = tensor.size(1)
    _fan_out = tensor.size(0)
    _fan_avg = int((_fan_in + _fan_out)/2)
    if mode == 'fan_in':
        n = _fan_in
    elif mode == 'fan_out':
        n = _fan_out
    elif mode == 'fan_avg':
        n = _fan_avg

    def _no_grad_uniform_(tensor, a, b):
        with torch.no_grad():
            return tensor.uniform_(a, b)

    def _no_grad_normal_(tensor, mean, std):
        with torch.no_grad():
            return tensor.normal_(mean, std)
    
    if distribution == 'normal':
        std = math.sqrt(scale / n)
        return _no_grad_normal_(tensor, mean=0, std=std)
    
    if distribution == 'uniform':
        limit = math.sqrt(3 * scale / n)
        return _no_grad_uniform_(tensor, -limit, limit)