import math

import torch
from torch import nn
from torch.nn import functional as F

from stylegan2.model import PixelNorm, EqualLinear, EqualConv2d
from stylegan2.op import conv2d_gradfix, upfirdn2d, fused_leaky_relu
from op import filtered_lrelu


def polyval(coef, x):
    res = 0

    for i, c in enumerate(coef):
        res += c * x ** (len(coef) - i - 1)

    return res


def bessel_j1(x):
    rp = [
        -8.99971225705559398224e8,
        4.52228297998194034323e11,
        -7.27494245221818276015e13,
        3.68295732863852883286e15,
    ]
    rq = [
        1.00000000000000000000e0,
        6.20836478118054335476e2,
        2.56987256757748830383e5,
        8.35146791431949253037e7,
        2.21511595479792499675e10,
        4.74914122079991414898e12,
        7.84369607876235854894e14,
        8.95222336184627338078e16,
        5.32278620332680085395e18,
    ]
    z1 = 1.46819706421238932572e1
    z2 = 4.92184563216946036703e1

    pp = [
        7.62125616208173112003e-4,
        7.31397056940917570436e-2,
        1.12719608129684925192e0,
        5.11207951146807644818e0,
        8.42404590141772420927e0,
        5.21451598682361504063e0,
        1.00000000000000000254e0,
    ]
    pq = [
        5.71323128072548699714e-4,
        6.88455908754495404082e-2,
        1.10514232634061696926e0,
        5.07386386128601488557e0,
        8.39985554327604159757e0,
        5.20982848682361821619e0,
        9.99999999999999997461e-1,
    ]
    qp = [
        5.10862594750176621635e-2,
        4.98213872951233449420e0,
        7.58238284132545283818e1,
        3.66779609360150777800e2,
        7.10856304998926107277e2,
        5.97489612400613639965e2,
        2.11688757100572135698e2,
        2.52070205858023719784e1,
    ]
    qq = [
        1.00000000000000000000e0,
        7.42373277035675149943e1,
        1.05644886038262816351e3,
        4.98641058337653607651e3,
        9.56231892404756170795e3,
        7.99704160447350683650e3,
        2.82619278517639096600e3,
        3.36093607810698293419e2,
    ]

    x = torch.as_tensor(x, dtype=torch.float64)

    z = x * x
    less5 = polyval(rp, z) / polyval(rq, z)
    less5 = less5 * x * (z - z1) * (z - z2)

    w = 5 / x
    z = w * w
    p = polyval(pp, z) / polyval(pq, z)
    q = polyval(qp, z) / polyval(qq, z)
    xn = x - (3 / 4 * math.pi)
    p = p * torch.cos(xn) - w * q * torch.sin(xn)
    more5 = p * math.sqrt(2 / math.pi) / torch.sqrt(x)

    y = torch.empty_like(x)
    flag = torch.abs(x) < 5
    y[flag] = less5[flag]
    y[~flag] = more5[~flag]

    return y


def jinc(x):
    pix = math.pi * x
    return 2 * bessel_j1(pix) / pix


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


def lowpass_filter(n_taps, cutoff, band_half, sr, use_jinc=False):
    window = kaiser_window(n_taps, band_half, sr)
    ind = torch.arange(n_taps) - (n_taps - 1) / 2

    if use_jinc:
        ind_sq = ind.unsqueeze(1) ** 2
        window = window.unsqueeze(1)
        coeff = jinc((2 * cutoff / sr) * torch.sqrt(ind_sq + ind_sq.T))
        lowpass = (2 * cutoff / sr) ** 2 * coeff * window * window.T
        lowpass = lowpass.to(torch.float32)

    else:
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
    def __init__(self, size, dim, sampling_rate, cutoff, eps=1e-8):
        super().__init__()

        freqs = torch.randn(dim, 2)
        radii = freqs.square().sum(1, keepdim=True).sqrt()
        freqs /= radii * radii.square().exp().pow(0.25)
        freqs *= cutoff
        phases = torch.rand(dim) - 0.5

        self.dim = dim
        self.size = torch.as_tensor(size).expand(2).tolist()
        self.sampling_rate = sampling_rate
        self.cutoff = cutoff

        self.register_buffer("freqs", freqs)
        self.register_buffer("phases", phases)

    def forward(self, batch_size, affine, transform=None):
        freqs = self.freqs.unsqueeze(0)
        phases = self.phases.unsqueeze(0)

        norm = torch.norm(affine[:, :2], dim=-1, keepdim=True)
        affine = affine / norm

        m_rot = torch.eye(3, device=affine.device).unsqueeze(0).repeat(batch_size, 1, 1)
        m_rot[:, 0, 0] = affine[:, 0]
        m_rot[:, 0, 1] = -affine[:, 1]
        m_rot[:, 1, 0] = affine[:, 1]
        m_rot[:, 1, 1] = affine[:, 0]

        m_tra = torch.eye(3, device=affine.device).unsqueeze(0).repeat(batch_size, 1, 1)
        m_tra[:, 0, 2] = -affine[:, 2]
        m_tra[:, 1, 2] = -affine[:, 3]

        if transform is not None:
            transform = m_rot @ m_tra @ transform

        else:
            transform = m_rot @ m_tra

        phases = phases + (freqs @ transform[:, :2, 2:]).squeeze(2)
        freqs = freqs @ transform[:, :2, :2]

        amplitude = (
            1
            - (freqs.norm(dim=2) - self.cutoff) / (self.sampling_rate / 2 - self.cutoff)
        ).clamp(0, 1)

        theta = torch.eye(2, 3, device=affine.device)
        theta[0, 0] = 0.5 * self.size[0] / self.sampling_rate
        theta[1, 1] = 0.5 * self.size[1] / self.sampling_rate
        grid = F.affine_grid(
            theta.unsqueeze(0), (1, 1, self.size[1], self.size[0]), align_corners=False
        )

        # x = (
        #     grid.unsqueeze(3) @ freqs.permute(0, 2, 1).unsqueeze(1).unsqueeze(2)
        # ).squeeze(3)
        x = (
            grid.unsqueeze(3) @ freqs.permute(0, 2, 1).view(-1, 1, 1, 2, self.dim)
        ).squeeze(3)

        x = x + phases.view(-1, 1, 1, self.dim)  # phases.unsqueeze(1).unsqueeze(2)
        x = torch.sin(x * (math.pi * 2))
        x = x * amplitude.view(
            -1, 1, 1, self.dim
        )  # amplitude.unsqueeze(1).unsqueeze(2)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        padding=0,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.padding = padding

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size})"

    def forward(self, input, style, input_gain=None, style_gain=None):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)

        if style_gain is not None:
            style = style * style_gain

        weight = self.weight

        # if self.demodulate:
        #     weight = weight * torch.rsqrt(weight.square().mean([2, 3, 4], keepdim=True))
        #     style = style * torch.rsqrt(style.square().mean())

        weight = weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.square().sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if input_gain is not None:
            weight = weight * input_gain

        input = input.view(1, batch * in_channel, height, width)
        out = conv2d_gradfix.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


class AliasFreeActivation(nn.Module):
    def __init__(
        self,
        out_channel,
        negative_slope,
        upsample_filter,
        downsample_filter,
        upsample,
        downsample,
        padding,
        clamp=None,
    ):
        super().__init__()

        self.register_buffer("upsample_filter", upsample_filter)
        self.register_buffer("downsample_filter", downsample_filter)

        self.negative_slope = negative_slope
        self.upsample = upsample
        self.downsample = downsample
        self.padding = padding
        self.clamp = clamp

    def forward(self, input, bias):
        out = filtered_lrelu(
            input,
            bias,
            self.upsample_filter,
            self.downsample_filter,
            self.upsample,
            self.downsample,
            self.padding,
            self.negative_slope,
            self.clamp,
        )

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
        padding=None,
        to_rgb=False,
        ema=0.999,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=demodulate,
            padding=kernel_size - 1,
        )

        self.bias = nn.Parameter(torch.zeros(out_channel))

        self.to_rgb = to_rgb
        if to_rgb:
            self.style_gain = 1 / (in_channel ** 0.5)

        else:
            self.style_gain = None

            self.activation = AliasFreeActivation(
                out_channel,
                0.2,
                upsample_filter,
                downsample_filter,
                upsample,
                2,
                padding=padding,
            )

        self.ema = ema
        self.register_buffer("ema_var", torch.tensor(1.0))

    def forward(self, input, style):
        if self.training:
            var = input.detach().square().mean()
            self.ema_var.mul_(self.ema).add_(var, alpha=1 - self.ema)

        out = self.conv(
            input,
            style,
            input_gain=torch.rsqrt(self.ema_var),
            style_gain=self.style_gain,
        )

        if self.to_rgb:
            out = out + self.bias.view(1, -1, 1, 1)

        else:
            out = self.activation(out, self.bias)

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
        ema=0.999,
        use_jinc=False,
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
        srs = filter_parameters["srs"]
        band_halfs = filter_parameters["band_halfs"]
        channels = filter_parameters["channels"]

        sizes = [sr + margin * 2 for sr in srs]
        sizes[-1] = srs[-1]

        self.input = FourierFeature(
            sizes[0], channels[0], sampling_rate=srs[0], cutoff=cutoffs[0]
        )
        self.affine_fourier = EqualLinear(style_dim, 4)
        self.affine_fourier.weight.detach().zero_()
        self.affine_fourier.bias.detach().copy_(
            torch.tensor([1, 0, 0, 0], dtype=torch.float32)
        )
        self.conv1 = EqualConv2d(channels[0], channels[0], 1, bias=False)

        self.convs = nn.ModuleList()
        for i in range(len(srs)):
            prev = max(i - 1, 0)

            mid_sr = max(srs[i], srs[prev]) * 2
            upsample = round(mid_sr / srs[prev])
            downsample = round(mid_sr / srs[i])

            up_filter = lowpass_filter(
                n_taps * upsample,
                cutoffs[prev],
                band_halfs[prev],
                mid_sr,
                use_jinc=use_jinc,
            )
            down_filter = lowpass_filter(
                n_taps * downsample,
                cutoffs[i],
                band_halfs[i],
                mid_sr,
                use_jinc=use_jinc,
            )

            in_size = torch.as_tensor(sizes[prev]).expand(2)
            out_size = torch.as_tensor(sizes[i]).expand(2)
            pad_total = (out_size - 1) * downsample + 1
            pad_total -= (in_size + kernel_size - 1) * upsample
            pad_total += n_taps * upsample + n_taps * downsample - 2
            pad_lo = (pad_total + upsample) // 2
            pad_hi = (pad_total - pad_lo).tolist()
            pad_lo = pad_lo.tolist()
            padding = [pad_lo[0], pad_hi[0], pad_lo[1], pad_hi[1]]

            self.convs.append(
                AliasFreeConv(
                    channels[prev],
                    channels[i],
                    kernel_size,
                    style_dim,
                    up_filter / up_filter.sum(),
                    down_filter / down_filter.sum(),
                    upsample,
                    padding=padding,
                    ema=ema,
                )
            )

        self.to_rgb = AliasFreeConv(
            channels[-1],
            3,
            1,
            style_dim,
            None,
            None,
            demodulate=False,
            to_rgb=True,
            ema=ema,
        )

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.conv1.weight.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, style, truncation=1, truncation_latent=None):
        latent = self.style(style)

        if truncation < 1:
            latent = truncation_latent + truncation * (latent - truncation_latent)

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

        out = self.to_rgb(out, latent) / 4

        return out
