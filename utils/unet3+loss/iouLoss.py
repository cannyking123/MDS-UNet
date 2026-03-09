# init_3d.py
import torch
import torch.nn as nn
from torch.nn import init

# ---- 判断是否是归一化层：默认权重=1，偏置=0 更稳 ----
_NORM_LAYERS = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.GroupNorm, nn.LayerNorm
)

def _init_norm(m):
    if hasattr(m, 'weight') and m.weight is not None:
        init.ones_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        init.zeros_(m.bias)

def weights_init_normal_3d(m, mean=0.0, std=0.02):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.ConvTranspose2d)):
        init.normal_(m.weight, mean, std)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, mean, std)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, _NORM_LAYERS):
        _init_norm(m)

def weights_init_xavier_3d(m, gain=1.0):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.ConvTranspose2d)):
        init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, _NORM_LAYERS):
        _init_norm(m)

def weights_init_kaiming_3d(m, a=0.0, mode='fan_in', nonlinearity='relu'):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, _NORM_LAYERS):
        _init_norm(m)

def weights_init_orthogonal_3d(m, gain=1.0):
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Conv2d, nn.ConvTranspose2d)):
        init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            init.zeros_(m.bias)
    elif isinstance(m, _NORM_LAYERS):
        _init_norm(m)

def init_weights_3d(net, init_type='kaiming', **kwargs):
    """
    init_type ∈ {'normal','xavier','kaiming','orthogonal'}
    例：init_weights_3d(model, 'kaiming', a=0.0, mode='fan_in', nonlinearity='relu')
    """
    if   init_type == 'normal':
        net.apply(lambda m: weights_init_normal_3d(m, **{k:v for k,v in kwargs.items() if k in ['mean','std']}))
    elif init_type == 'xavier':
        net.apply(lambda m: weights_init_xavier_3d(m, **{k:v for k,v in kwargs.items() if k in ['gain']}))
    elif init_type == 'kaiming':
        net.apply(lambda m: weights_init_kaiming_3d(m, **{k:v for k,v in kwargs.items() if k in ['a','mode','nonlinearity']}))
    elif init_type == 'orthogonal':
        net.apply(lambda m: weights_init_orthogonal_3d(m, **{k:v for k,v in kwargs.items() if k in ['gain']}))
    else:
        raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
