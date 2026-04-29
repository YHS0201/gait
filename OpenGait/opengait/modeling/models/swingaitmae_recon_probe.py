import torch.optim as optim

from .swingaitmae import SwinGaitmae
from utils import get_attr_from, get_valid_args


class SwinGaitmaeFrozenEncoderReconProbe(SwinGaitmae):
    """Freeze encoder and ID heads, and train shallow reconstruction decoders only."""

    def build_network(self, model_cfg):
        super().build_network(model_cfg)
        self._frozen_modules = [
            self.layer0,
            self.layer1,
            self.layer2,
            self.ulayer,
            self.transformer,
            self.FCs,
            self.BNNecks,
        ]
        for module in self._frozen_modules:
            module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        for module in getattr(self, '_frozen_modules', []):
            module.eval()
        return self

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        optimizer = get_attr_from([optim], optimizer_cfg['solver'])
        get_valid_args(optimizer, optimizer_cfg, ['solver'])

        params_list = [
            {'params': self.decoder.parameters(), 'lr': optimizer_cfg['lr'], 'weight_decay': optimizer_cfg['weight_decay']},
            {'params': self.decoder_edge.parameters(), 'lr': optimizer_cfg['lr'], 'weight_decay': optimizer_cfg['weight_decay']},
        ]
        return optimizer(params_list)

    def forward(self, inputs):
        retval = super().forward(inputs)
        if self.training:
            retval['training_feat'] = {
                key: value
                for key, value in retval['training_feat'].items()
                if not isinstance(value, dict)
            }
        return retval