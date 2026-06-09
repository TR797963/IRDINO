'''Copyright (c) 2024 The DEIM Authors. All Rights Reserved.'''
import torch
import torch.nn as nn
from ..core import register
__all__ = ['DEIM']

@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'sftb_adapter', 'smahe']

    def __init__(self, backbone: nn.Module, encoder: nn.Module, decoder: nn.Module, sftb_adapter: nn.Module=None, smahe: nn.Module=None, freeze_detector_stats: bool=False):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.sftb_adapter = sftb_adapter
        self.smahe = smahe
        self.freeze_detector_stats = freeze_detector_stats

    def train(self, mode: bool=True):
        super().train(mode)
        if mode and self.freeze_detector_stats:
            for detector_module in (self.backbone, self.encoder, self.decoder):
                for layer in detector_module.modules():
                    if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                        layer.eval()
        return self

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        losses = None
        diagnostics = None
        if self.sftb_adapter is not None:
            x, sftb_losses, sftb_diag = self.sftb_adapter(x, targets=targets)
            if sftb_losses:
                losses = {**(losses or {}), **sftb_losses}
            if sftb_diag:
                diagnostics = {**(diagnostics or {}), **{f'sftb_{k}': v for k, v in sftb_diag.items()}}
        if self.smahe is not None:
            x, smahe_losses, smahe_diag = self.smahe(x, targets=targets)
            if smahe_losses:
                losses = {**(losses or {}), **smahe_losses}
            if smahe_diag:
                diagnostics = {**(diagnostics or {}), **{f'smahe_{k}': v for k, v in smahe_diag.items()}}
        x = self.decoder(x, targets)
        if losses:
            x.update(losses)
            x['loss_irdino'] = torch.stack(list(losses.values())).sum()
        if diagnostics:
            x['irdino_diagnostics'] = diagnostics
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
