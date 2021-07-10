import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from tensorfn import load_arg_config, distributed as dist, get_logger

try:
    import wandb

except ImportError:
    wandb = None


from stylegan2.dataset import MultiResolutionDataset
from stylegan2.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from stylegan2.op import conv2d_gradfix
from stylegan2.non_leaking import augment, AdaptiveAugment
from stylegan2.model import Discriminator
from config import GANConfig
from model import Generator, filter_parameters


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    buf1 = dict(model1.named_buffers())
    buf2 = dict(model2.named_buffers())

    for k in buf1.keys():
        if "ema_var" not in k:
            continue

        buf1[k].data.mul_(0).add_(buf2[k].data, alpha=1)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(conf, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    loader = sample_data(loader)

    pbar = range(conf.training.iter)

    if get_rank() == 0:
        pbar = tqdm(
            pbar, initial=conf.training.start_iter, dynamic_ncols=True, smoothing=0.01
        )

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if conf.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = conf.training.augment_p if conf.training.augment_p > 0 else 0.0
    r_t_stat = 0

    if conf.training.augment and conf.training.augment_p == 0:
        ada_augment = AdaptiveAugment(
            conf.training.ada_target, conf.training.ada_length, 8, device
        )

    sample_z = torch.randn(
        conf.training.n_sample, conf.generator["style_dim"], device=device
    )

    for idx in pbar:
        i = idx + conf.training.start_iter

        if i > conf.training.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = make_noise(conf.training.batch, conf.generator["style_dim"], 1, device)
        generator.eval()
        fake_img = generator(noise)
        generator.train()

        if conf.training.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if conf.training.augment and conf.training.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % conf.training.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if conf.training.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (
                conf.training.r1 / 2 * r1_loss * conf.training.d_reg_every
                + 0 * real_pred[0]
            ).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = make_noise(conf.training.batch, conf.generator["style_dim"], 1, device)
        fake_img = generator(noise)

        if conf.training.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and conf.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                    }
                )

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample = g_ema(sample_z)
                    utils.save_image(
                        sample,
                        f"sample/{str(i).zfill(6)}.png",
                        nrow=int(conf.training.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

            if i % 10000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "conf": conf.dict(),
                        "ada_aug_p": ada_aug_p,
                    },
                    f"checkpoint/{str(i).zfill(6)}.pt",
                )


def main(conf):
    device = "cuda"
    conf.distributed = conf.n_gpu > 1

    logger = get_logger(mode=conf.logger)
    logger.info(conf.dict())

    generator = conf.generator.make().to(device)
    g_ema = conf.generator.make().to(device)
    discriminator = conf.discriminator.make().to(device)
    accumulate(g_ema, generator, 0)

    logger.info(generator)
    logger.info(discriminator)

    d_reg_ratio = conf.training.d_reg_every / (conf.training.d_reg_every + 1)

    g_optim = optim.Adam(generator.parameters(), lr=conf.training.lr_g, betas=(0, 0.99))
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=conf.training.lr_d * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if conf.ckpt is not None:
        logger.info(f"load model: {conf.ckpt}")

        ckpt = torch.load(conf.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(conf.ckpt)
            conf.training.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    if conf.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
            broadcast_buffers=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(conf.path, transform, conf.training.size)
    loader = data.DataLoader(
        dataset,
        batch_size=conf.training.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=conf.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and conf.wandb:
        wandb.init(project="alias free gan")

    train(conf, loader, generator, discriminator, g_optim, d_optim, g_ema, device)


if __name__ == "__main__":
    conf = load_arg_config(GANConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
