import torch.nn as nn

from .swingaitmae import SwinGaitmae


class _StopGradInputWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x.detach())


class SwinGaitmaeG2StopGrad(SwinGaitmae):
    """G2 experiment: keep ID + reconstruction, but stop reconstruction gradients at decoder input."""

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self.decoder = _StopGradInputWrapper(self.decoder)
        self.decoder_edge = _StopGradInputWrapper(self.decoder_edge)