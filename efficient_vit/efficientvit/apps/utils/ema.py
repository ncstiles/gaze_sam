# EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction
# Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han
# International Conference on Computer Vision (ICCV), 2023

import copy
import math

import torch
import torch.nn as nn

from efficient_vit.efficientvit.models.utils import is_parallel

__all__ = ["EMA"]


def update_ema(ema, new_state_dict, decay):
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v -= (1.0 - decay) * (v - new_state_dict[k].detach())


class EMA:
    def __init__(self, model, decay, warmup_steps=2000):
        self.shadows = copy.deepcopy(model.module if is_parallel(model) else model).eval()
        self.decay = decay
        self.warmup_steps = warmup_steps

        for p in self.shadows.parameters():
            p.requires_grad = False

    def step(self, model, global_step):
        with torch.no_grad():
            msd = (model.module if is_parallel(model) else model).state_dict()
            update_ema(self.shadows, msd, self.decay * (1 - math.exp(-global_step / self.warmup_steps)))

    def state_dict(self):
        return {self.decay: self.shadows.state_dict()}

    def load_state_dict(self, state_dict):
        for decay in state_dict:
            if decay == self.decay:
                self.shadows.load_state_dict(state_dict[decay])
