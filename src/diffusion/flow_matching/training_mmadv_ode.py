import random

import torch
import copy
import timm
import torchvision.transforms.v2.functional
from torch.nn import Parameter

from src.utils.no_grad import no_grad
from typing import Callable, Iterator, Tuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler
from src.diffusion.base.sampling import BaseSampler

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

from PIL import Image
import numpy as np

def time_shift_fn(t, timeshift=1.0):
    return t/(t+(1-t)*timeshift)

def random_crop(images, resize, crop_size):
    images = torchvision.transforms.v2.functional.resize(images, size=resize, antialias=True)
    h, w = crop_size
    h0 = random.randint(0, images.shape[2]-h)
    w0 = random.randint(0, images.shape[3]-w)
    return images[:, :, h0:h0+h, w0:w0+w]

# class EulerSolver:
#     def __init__(
#             self,
#             num_steps: int,
#             *args,
#             **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.num_steps = num_steps
#         self.timesteps = torch.linspace(0.0, 1, self.num_steps+1, dtype=torch.float32)
#
#     def __call__(self, net, noise, timeshift, condition):
#         steps = time_shift_fn(self.timesteps[:, None], timeshift[None, :]).to(noise.device, noise.dtype)
#         x = noise
#         trajs = [x, ]
#         for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
#             dt = t_next - t_cur
#             v = net(x, t_cur, condition)
#             x = x + v*dt[:, None, None, None]
#             x = x.to(noise.dtype)
#             trajs.append(x)
#         return trajs
#
# class NeuralSolver(nn.Module):
#     def __init__(
#             self,
#             num_steps: int,
#             *args,
#             **kwargs
#     ):
#         super().__init__(*args, **kwargs)
#         self.num_steps = num_steps
#         self.timedeltas = torch.nn.Parameter(torch.ones((num_steps))/num_steps, requires_grad=True)
#         self.coeffs = torch.nn.Parameter(torch.zeros((num_steps, num_steps)), requires_grad=True)
#         # self.golden_noise = torch.nn.Parameter(torch.randn((1, 3, 1024, 1024))*0.01, requires_grad=True)
#
#     def forward(self, net, noise, timeshift, condition):
#         batch_size, c, height, width = noise.shape
#         # golden_noise = torch.nn.functional.interpolate(self.golden_noise, size=(height, width), mode='bicubic', align_corners=False)
#         x = noise # + golden_noise.repeat(batch_size, 1, 1, 1)
#         x_trajs = [x, ]
#         v_trajs = []
#         dts = self.timedeltas.softmax(dim=0)
#         print(dts)
#         coeffs = self.coeffs
#         t_cur = torch.zeros((batch_size,), dtype=noise.dtype, device=noise.device)
#         for i, dt in enumerate(dts):
#             pred_v = net(x, t_cur, condition)
#             v = torch.zeros_like(pred_v)
#             v_trajs.append(pred_v)
#             acc_coeffs = 0.0
#             for j in range(i):
#                 acc_coeffs = acc_coeffs + coeffs[i, j]
#                 v = v + coeffs[i, j]*v_trajs[j]
#             v = v + (1-acc_coeffs)*v_trajs[i]
#             x = x + v*dt
#             x = x.to(noise.dtype)
#             x_trajs.append(x)
#             t_cur = t_cur +  dt
#         return x_trajs

import re
import os
import unicodedata
def clean_filename(s):
    # 去除首尾空格和点号
    s = s.strip().strip('.')
    # 转换 Unicode 字符为 ASCII 形式
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
    illegal_chars = r'[/]'
    reserved_names = set()
    # 替换非法字符为下划线
    s = re.sub(illegal_chars, '_', s)
    # 合并连续的下划线
    s = re.sub(r'_{2,}', '_', s)
    # 转换为小写
    s = s.lower()
    # 检查是否为保留文件名
    if s.upper() in reserved_names:
        s = s + '_'
    # 限制文件名长度
    max_length = 200
    s = s[:max_length]
    if not s:
        return 'untitled'
    return s

def prompt_augment(prompts, random_prompts, replace_prob=0.5, front_append_prob=0.5, back_append_prob=0.5, delete_prob=0.5,):
    random_prompts = random.choices(random_prompts, k=len(prompts))
    new_prompts = []
    for prompt, random_prompt in zip(prompts, random_prompts):
        if random.random() < replace_prob:
            new_prompt = random_prompt
        else:
            new_prompt = prompt
        if random.random() < front_append_prob:
            new_prompt = random_prompt + ", " + new_prompt
        if random.random() < back_append_prob:
            new_prompt = new_prompt + ", " + random_prompt
        if random.random() < delete_prob:
            new_length = random.randint(1, len(new_prompt.split(",")))
            new_prompt = ", ".join(new_prompt.split(",")[:new_length])
        new_prompts.append(new_prompt)
    return new_prompts

class AdvODETrainer(BaseTrainer):
    def __init__(
            self,
            scheduler: BaseScheduler,
            loss_weight_fn:Callable=constant,
            adv_loss_weight: float=0.5,
            gan_loss_weight: float=0.5,
            im_encoder:nn.Module=None,
            mm_encoder:nn.Module=None,
            adv_head:nn.Module=None,
            random_crop_size=448,
            max_image_size=512,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn
        self.adv_loss_weight = adv_loss_weight
        self.gan_loss_weight = gan_loss_weight
        self.im_encoder = im_encoder
        self.mm_encoder = mm_encoder
        self.adv_head = adv_head

        self.real_buffer = []
        self.fake_buffer = []
        self.random_crop_size = random_crop_size
        self.max_image_size = max_image_size

        no_grad(self.im_encoder)
        no_grad(self.mm_encoder)
        self.random_prompts = ["hahahaha", ]
        self.saved_filenames = []

    def preproprocess(self, x, condition, uncondition, metadata):
        self.uncondition = uncondition
        return super().preproprocess(x, condition, uncondition, metadata)

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        batch_size, c, height, width = x.shape
        noise = torch.randn_like(x)
        _, trajs = solver(net, noise, y, self.uncondition, return_x_trajs=True, return_v_trajs=False)
        with torch.no_grad():
           _, ref_trajs = solver(ema_net, noise, y, self.uncondition, return_x_trajs=True, return_v_trajs=False)

        fake_x0 = (trajs[-1]+1)/2
        fake_x0 = fake_x0.clamp(0, 1)
        prompt = metadata["prompt"]
        self.random_prompts.extend(prompt)
        self.random_prompts = self.random_prompts[-50:]
        filename = clean_filename(prompt[0])+".png"
        Image.fromarray((fake_x0[0].permute(1, 2, 0).detach().cpu().float() * 255).to(torch.uint8).numpy()).save(f'{filename}')
        self.saved_filenames.append(filename)
        if len(self.saved_filenames) > 100:
            os.remove(self.saved_filenames[0])
            self.saved_filenames.pop(0)

        real_x0 = metadata["raw_image"]
        fake_x0 = random_crop(fake_x0, resize=self.max_image_size, crop_size=(self.random_crop_size, self.random_crop_size))
        real_x0 = random_crop(real_x0, resize=self.max_image_size, crop_size=(self.random_crop_size, self.random_crop_size))

        fake_im_features = self.im_encoder(fake_x0, resize=False)
        fake_mm_features = self.mm_encoder(fake_x0, prompt, resize=True)
        fake_im_features_detach = fake_im_features.detach()
        fake_mm_features_detach = fake_mm_features.detach()

        with torch.no_grad():
            real_im_features = self.im_encoder(real_x0, resize=False)
            real_mm_features = self.mm_encoder(real_x0, prompt, resize=True)
            not_match_prompt = prompt_augment(prompt, self.random_prompts)#random.choices(self.random_prompts, k=batch_size)
            real_not_match_mm_features = self.mm_encoder(real_x0, not_match_prompt, resize=True)
            self.real_buffer.append((real_im_features, real_mm_features))
            self.fake_buffer.append((fake_im_features_detach, fake_mm_features_detach))
            self.fake_buffer.append((real_im_features, real_not_match_mm_features))
            while len(self.real_buffer) > 10:
                self.real_buffer.pop(0)
            while len(self.fake_buffer) > 10:
                self.fake_buffer.pop(0)

        real_features_gan = torch.cat([x[0] for x in self.real_buffer], dim=0)
        real_conditions_gan = torch.cat([x[1] for x in self.real_buffer], dim=0)
        fake_features_gan = torch.cat([x[0] for x in self.fake_buffer], dim=0)
        fake_conditions_gan = torch.cat([x[1] for x in self.fake_buffer], dim=0)
        real_score_gan = self.adv_head(real_features_gan, real_conditions_gan)
        fake_score_gan = self.adv_head(fake_features_gan, fake_conditions_gan)

        fake_score_adv = self.adv_head(fake_im_features, fake_mm_features)
        fake_score_detach_adv = self.adv_head(fake_im_features_detach, fake_mm_features_detach)


        loss_gan = -torch.log(1 - fake_score_gan).mean() - torch.log(real_score_gan).mean()
        acc_real = (real_score_gan > 0.5).float()
        acc_fake = (fake_score_gan < 0.5).float()
        loss_adv = -torch.log(fake_score_adv)
        loss_adv_hack = torch.log(fake_score_detach_adv)

        trajs_loss = 0.0
        for x_t, ref_x_t in zip(trajs, ref_trajs):
            trajs_loss = trajs_loss + torch.abs(x_t - ref_x_t).mean()
        trajs_loss = trajs_loss / len(trajs)

        out = dict(
            trajs_loss=trajs_loss.mean(),
            adv_loss=loss_adv.mean(),
            gan_loss=loss_gan.mean(),
            acc_real=acc_real.mean(),
            acc_fake=acc_fake.mean(),
            loss=trajs_loss.mean() + self.adv_loss_weight*(loss_adv.mean() + loss_adv_hack.mean())+self.gan_loss_weight*loss_gan.mean(),
        )
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        self.adv_head.state_dict(
            destination=destination,
            prefix=prefix + "adv_head.",
            keep_vars=keep_vars)
