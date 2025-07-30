import torch
import copy
import timm
from torch.nn import Parameter

from src.utils.no_grad import no_grad
from typing import Callable, Iterator, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler

def inverse_sigma(alpha, sigma):
    return 1/sigma**2
def snr(alpha, sigma):
    return alpha/sigma
def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, min=threshold)
def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha/sigma, max=threshold)
def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)


class DisperseTrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            feat_loss_weight: float=0.5,
            lognorm_t=False,
            timeshift=1.0,
            align_layer=8,
            temperature=1.0,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lognorm_t = lognorm_t
        self.scheduler = scheduler
        self.timeshift = timeshift
        self.loss_weight_fn = loss_weight_fn
        self.feat_loss_weight = feat_loss_weight
        self.align_layer = align_layer
        self.temperature = temperature

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        batch_size, c, height, width = x.shape
        if self.lognorm_t:
            base_t = torch.randn((batch_size), device=x.device, dtype=torch.float32).sigmoid()
        else:
            base_t = torch.rand((batch_size), device=x.device, dtype=torch.float32)
        t = time_shift_fn(base_t, self.timeshift).to(x.dtype)
        noise = torch.randn_like(x)
        alpha = self.scheduler.alpha(t)
        dalpha = self.scheduler.dalpha(t)
        sigma = self.scheduler.sigma(t)
        dsigma = self.scheduler.dsigma(t)

        x_t = alpha * x + noise * sigma
        v_t = dalpha * x + dsigma * noise

        src_feature = []
        def forward_hook(net, input, output):
            feature = output
            if isinstance(feature, tuple):
                feature = feature[0] # mmdit
            src_feature.append(feature)

        if getattr(net, "encoder", None) is not None:
            handle = net.encoder.blocks[self.align_layer - 1].register_forward_hook(forward_hook)
        else:
            handle = net.blocks[self.align_layer - 1].register_forward_hook(forward_hook)

        out = net(x_t, t, y)
        handle.remove()
        disperse_distance = 0.0
        for sf in src_feature:
            sf = torch.mean(sf, dim=1, keepdim=False)
            distance = (sf[None, :, :] - sf[:, None, :])**2
            distance = distance.sum(dim=-1)
            sf_disperse_loss = torch.exp(-distance/self.temperature)
            mask = 1-torch.eye(batch_size, device=distance.device, dtype=distance.dtype)
            disperse_distance += (sf_disperse_loss*mask).sum()/mask.numel() + 1e-6
        disperse_loss = disperse_distance.log()


        weight = self.loss_weight_fn(alpha, sigma)
        fm_loss = weight*(out - v_t)**2

        out = dict(
            fm_loss=fm_loss.mean(),
            cos_loss=disperse_loss.mean(),
            loss=fm_loss.mean() + self.feat_loss_weight*disperse_loss.mean(),
        )
        return out

