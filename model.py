import math

import torch
from torch import nn
from torch.nn import functional as F

from stylegan2.model import PixelNorm, EqualLinear, EqualConv2d
from stylegan2.op import conv2d_gradfix, upfirdn2d, fused_leaky_relu


def kaiser_attenuation(n_taps, f_h, sr):
    df = (2 * f_h) / (sr / 2)

    return 2.285 * (n_taps - 1) * math.pi * df + 7.95


def kaiser_beta(n_taps, f_h, sr):
    atten = kaiser_attenuation(n_taps, f_h, sr)

    if atten > 50:
        return 0.1102 * (atten - 8.7)

    elif 50 >= atten >= 21:
        return 0.5842 * (atten - 21) ** 0.4 + 0.07886 * (atten - 21)

    else:
        return 0.0


def kaiser_window(n_taps, f_h, sr):
    beta = kaiser_beta(n_taps, f_h, sr)
    ind = torch.arange(n_taps) - (n_taps - 1) / 2

    return torch.i0(beta * torch.sqrt(1 - ((2 * ind) / (n_taps - 1)) ** 2)) / torch.i0(
        torch.tensor(beta)
    )


def lowpass_filter(n_taps, cutoff, band_half, sr):
    window = kaiser_window(n_taps, band_half, sr)
    ind = torch.arange(n_taps) - (n_taps - 1) / 2
    lowpass = 2 * cutoff / sr * torch.sinc(2 * cutoff / sr * ind) * window

    return lowpass


def filter_parameters(
    n_layer,
    n_critical,
    sr_max,
    cutoff_0,
    cutoff_n,
    stopband_0,
    stopband_n,
    channel_max,
    channel_base,
):
    cutoffs = []
    stopbands = []
    srs = []
    band_halfs = []
    channels = []

    for i in range(n_layer):
        f_c = cutoff_0 * (cutoff_n / cutoff_0) ** min(i / (n_layer - n_critical), 1)
        f_t = stopband_0 * (stopband_n / stopband_0) ** min(
            i / (n_layer - n_critical), 1
        )
        s_i = 2 ** math.ceil(math.log(min(2 * f_t, sr_max), 2))
        f_h = max(f_t, s_i / 2) - f_c
        c_i = min(round(channel_base / s_i), channel_max)

        cutoffs.append(f_c)
        stopbands.append(f_t)
        srs.append(s_i)
        band_halfs.append(f_h)
        channels.append(c_i)

    return {
        "cutoffs": cutoffs,
        "stopbands": stopbands,
        "srs": srs,
        "band_halfs": band_halfs,
        "channels": channels,
    }


class FourierFeature(nn.Module):
    def __init__(self, size, dim, cutoff, eps=1e-8):
        super().__init__()

        coords = torch.linspace(-1, 1, size + 1)[:-1]
        freqs = torch.linspace(0, cutoff, dim // 4)

        self.register_buffer("coords", coords)
        self.register_buffer("freqs", freqs)
        self.register_buffer(
            "lf", freqs.view(1, dim // 4, 1, 1) * 2 * math.pi * 2 / size
        )
        self.eps = eps

    def forward(self, batch_size, affine=None):
        coord_map = torch.ger(self.freqs, self.coords)
        coord_map = 2 * math.pi * coord_map
        size = self.coords.shape[0]
        coord_h = coord_map.view(self.freqs.shape[0], 1, size)
        coord_w = coord_h.transpose(1, 2)

        if affine is not None:
            norm = torch.norm(affine[:, :2], dim=-1, keepdim=True)
            affine = affine / (norm + self.eps)

            r_c, r_s, t_x, t_y = affine.view(
                affine.shape[0], 1, 1, 1, affine.shape[-1]
            ).unbind(-1)

            coord_h_orig = coord_h.unsqueeze(0)
            coord_w_orig = coord_w.unsqueeze(0)

            coord_h = -coord_w_orig * r_s + coord_h_orig * r_c - t_y * self.lf
            coord_w = coord_w_orig * r_c + coord_h_orig * r_s - t_x * self.lf

            coord_h = torch.cat((torch.sin(coord_h), torch.cos(coord_h)), 1)
            coord_w = torch.cat((torch.sin(coord_w), torch.cos(coord_w)), 1)

            coord_h = coord_h.expand(-1, -1, size, -1)
            coord_w = coord_w.expand(-1, -1, -1, size)
            coords = torch.cat((coord_h, coord_w), 1)

            return coords

        else:
            coord_h = torch.cat((torch.sin(coord_h), torch.cos(coord_h)), 0)
            coord_w = torch.cat((torch.sin(coord_w), torch.cos(coord_w)), 0)

            coord_h = coord_h.expand(-1, size, -1)
            coord_w = coord_w.expand(-1, -1, size)
            coords = torch.cat((coord_h, coord_w), 0)

            return coords.unsqueeze(0).expand(batch_size, -1, -1, -1)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        decay=0.9989,
        padding=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)

        if padding:
            self.padding = kernel_size // 2

        else:
            self.padding = 0

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.register_buffer("ema_var", torch.tensor(1.0))
        self.decay = decay

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size})"

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.training:
            var = input.pow(2).mean((0, 1, 2, 3))
            self.ema_var.mul_(self.decay).add_(var.detach(), alpha=1 - self.decay)

        weight = weight / (torch.sqrt(self.ema_var) + 1e-8)

        input = input.view(1, batch * in_channel, height, width)
        out = conv2d_gradfix.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


def upsample(x, kernel, factor, pad=(0, 0)):
    x = upfirdn2d(x, kernel.unsqueeze(0), up=(factor, 1), pad=(*pad, 0, 0))
    x = upfirdn2d(x, kernel.unsqueeze(1), up=(1, factor), pad=(0, 0, *pad))

    return x


def downsample(x, kernel, factor, pad=(0, 0)):
    x = upfirdn2d(x, kernel.unsqueeze(0), down=(factor, 1), pad=(*pad, 0, 0))
    x = upfirdn2d(x, kernel.unsqueeze(1), down=(1, factor), pad=(0, 0, *pad))

    return x


class AliasFreeActivation(nn.Module):
    def __init__(
        self,
        out_channel,
        negative_slope,
        upsample_filter,
        downsample_filter,
        upsample,
        downsample,
        margin,
    ):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(out_channel))
        self.register_buffer("upsample_filter", upsample_filter * upsample)
        self.register_buffer("downsample_filter", downsample_filter)

        self.negative_slope = negative_slope
        self.upsample = upsample
        self.downsample = downsample
        self.margin = margin

        p = upsample_filter.shape[0] - upsample
        self.up_pad = ((p + 1) // 2 + upsample - 1, p // 2)

        p = downsample_filter.shape[0] - downsample
        self.down_pad = ((p + 1) // 2, p // 2)

    def forward(self, input):
        out = input + self.bias.view(1, -1, 1, 1)
        out = upsample(out, self.upsample_filter, self.upsample, pad=self.up_pad)
        out = fused_leaky_relu(out, negative_slope=self.negative_slope)
        out = downsample(
            out, self.downsample_filter, self.downsample, pad=self.down_pad
        )
        m = self.margin
        m = m * self.upsample - m * self.downsample
        m //= 2

        if m > 0:
            out = out[:, :, m:-m, m:-m]

        return out


class AliasFreeConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample_filter,
        downsample_filter,
        upsample=1,
        demodulate=True,
        margin=10,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel, out_channel, kernel_size, style_dim, demodulate=demodulate
        )

        self.activation = AliasFreeActivation(
            out_channel,
            0.2,
            upsample_filter,
            downsample_filter,
            upsample * 2,
            2,
            margin=margin,
        )

    def forward(self, input, style):
        out = self.conv(input, style)
        out = self.activation(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(3))

    def forward(self, input, style):
        out = self.conv(input, style)
        out = out + self.bias.view(1, -1, 1, 1)

        return out


class Generator(nn.Module):
    def __init__(
        self,
        style_dim,
        n_mlp,
        kernel_size,
        n_taps,
        filter_parameters,
        margin=10,
        lr_mlp=0.01,
    ):
        super().__init__()

        self.style_dim = style_dim
        self.margin = margin

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        cutoffs = filter_parameters["cutoffs"]
        stopbands = filter_parameters["stopbands"]
        srs = filter_parameters["srs"]
        band_halfs = filter_parameters["band_halfs"]
        channels = filter_parameters["channels"]

        self.input = FourierFeature(srs[0] + margin * 2, channels[0], cutoff=cutoffs[0])
        self.affine_fourier = EqualLinear(style_dim, 4)
        self.affine_fourier.weight.detach().zero_()
        self.affine_fourier.bias.detach().copy_(
            torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        )
        self.conv1 = EqualConv2d(channels[0], channels[0], 1)

        self.convs = nn.ModuleList()
        for i in range(len(srs)):
            prev = max(i - 1, 0)
            sr = srs[i]

            up = 1
            if srs[prev] < sr:
                up = 2

            up_filter = lowpass_filter(
                n_taps * up * 2, cutoffs[prev], band_halfs[prev], srs[i] * up * 2
            )
            down_filter = lowpass_filter(
                n_taps * up, cutoffs[i], band_halfs[i], srs[i] * up * 2
            )

            self.convs.append(
                AliasFreeConv(
                    channels[prev],
                    channels[i],
                    kernel_size,
                    style_dim,
                    up_filter / up_filter.sum(),
                    down_filter / down_filter.sum(),
                    up,
                    margin=margin,
                )
            )

        self.to_rgb = ToRGB(channels[-1], style_dim)

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.conv1.weight.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_transform(self, style, truncation=1, truncation_latent=None):
        latent = self.style(style)

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        return self.affine_fourier(latent)

    def forward(self, style, truncation=1, truncation_latent=None, transform=None):
        latent = self.style(style)

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

        if transform is None:
            transform = self.affine_fourier(latent)

        out = self.input(latent.shape[0], transform)
        out = self.conv1(out)

        for conv in self.convs:
            out = conv(out, latent)

        out = out[:, :, self.margin : -self.margin, self.margin : -self.margin]
        out = self.to_rgb(out, latent) / 4

        return out
