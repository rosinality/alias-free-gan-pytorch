# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from collections import abc

import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

from stylegan2.op import upfirdn2d, fused_leaky_relu

module_path = os.path.dirname(__file__)
filtered_lrelu_op = load(
    "filtered_lrelu",
    sources=[
        os.path.join(module_path, "filtered_lrelu.cpp"),
        os.path.join(module_path, "filtered_lrelu_wr.cu"),
        os.path.join(module_path, "filtered_lrelu_rd.cu"),
        os.path.join(module_path, "filtered_lrelu_ns.cu"),
        # os.path.join(module_path, "filtered_lrelu.h"),
        os.path.join(module_path, "filtered_lrelu.cu"),
    ],
    extra_cuda_cflags=["--use_fast_math"],
)


def format_padding(padding):
    if not isinstance(padding, abc.Iterable):
        padding = (padding, padding)

    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])

    return padding


def filtered_lrelu(
    x, bias, up_filter, down_filter, up, down, padding, negative_slope=0.2, clamp=None
):
    padding = format_padding(padding)

    if x.device.type == "cuda":
        return FilteredLReLU.apply(
            x,
            bias,
            up_filter,
            down_filter,
            up,
            down,
            padding,
            negative_slope,
            2 ** 0.5,
            clamp,
            False,
            None,
            0,
            0,
        )

    return filtered_lrelu_upfirdn2d(
        x, bias, up_filter, down_filter, up, down, padding, negative_slope, clamp
    )


def filtered_lrelu_upfirdn2d(
    x, bias, up_filter, down_filter, up, down, padding, negative_slope=0.2, clamp=None
):
    if bias is not None:
        x = x + bias.view(1, -1, 1, 1)

    x = upfirdn2d(x, up_filter, up=up, pad=padding, gain=up ** 2)
    x = fused_leaky_relu(x, negative_slope=negative_slope)

    if clamp is not None:
        x = x.clamp(-clamp, clamp)

    x = upfirdn2d(x, down_filter, down=down)

    return x


class FilteredLReLU(Function):
    @staticmethod
    def forward(
        ctx,
        x,
        bias,
        up_filter,
        down_filter,
        up,
        down,
        padding,
        negative_slope,
        gain,
        clamp,
        flip_filter,
        sign,
        sign_offset_x,
        sign_offset_y,
    ):
        if up_filter is None:
            up_filter = torch.ones([1, 1], dtype=torch.float32, device=x.device)

        if down_filter is None:
            down_filter = torch.ones([1, 1], dtype=torch.float32, device=x.device)

        if up == 1 and up_filter.ndim == 1 and up_filter.shape[0] == 1:
            up_filter = up_filter.square()[None]

        if down == 1 and down_filter.ndim == 1 and down_filter.shape[0] == 1:
            down_filter = down_filter.square()[None]

        clamp = float(clamp if clamp is not None else "inf")

        if sign is None:
            sign = torch.empty([0])

        if bias is None:
            bias = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)

        write_signs = (sign.numel() == 0) and (x.requires_grad or bias.requires_grad)

        # strides = [x.stride(i) for i in range(x.ndim) if x.shape[i] > 1]
        # if any(a < b for a, b in zip(strides[:-1], strides[1:])):

        pad_x0, pad_x1, pad_y0, pad_y1 = padding

        if x.dtype in (torch.float16, torch.float32):
            # if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):

            y, sign_out, return_code = filtered_lrelu_op.filtered_lrelu(
                x,
                up_filter,
                down_filter,
                bias,
                sign,
                up,
                down,
                pad_x0,
                pad_x1,
                pad_y0,
                pad_y1,
                sign_offset_x,
                sign_offset_y,
                gain,
                negative_slope,
                clamp,
                flip_filter,
                write_signs,
            )

        else:
            return_code = -1

        if return_code < 0:
            y = x + bias.view(-1, 1, 1)
            y = upfirdn2d(
                y, up_filter, up=up, pad=padding, flip_filter=flip_filter, gain=up ** 2
            )
            sign_out = filtered_lrelu_op.filtered_lrelu_act_(
                y,
                sign,
                sign_offset_x,
                sign_offset_y,
                gain,
                negative_slope,
                clamp,
                write_signs,
            )
            y = upfirdn2d(y, down_filter, down=down, flip_filter=flip_filter)

        ctx.save_for_backward(
            up_filter, down_filter, (sign if sign.numel() else sign_out)
        )
        ctx.x_shape = x.shape
        ctx.y_shape = y.shape
        ctx.sign_offsets = sign_offset_x, sign_offset_y
        ctx.padding = padding
        ctx.args = up, down, negative_slope, gain, flip_filter

        return y

    @staticmethod
    def backward(ctx, dy):
        up_filter, down_filter, sign = ctx.saved_tensors
        _, _, x_h, x_w = ctx.x_shape
        _, _, y_h, y_w = ctx.y_shape
        sign_offset_x, sign_offset_y = ctx.sign_offsets
        pad_x0, pad_x1, pad_y0, pad_y1 = ctx.padding
        up, down, negative_slope, gain, flip_filter = ctx.args

        dx = None
        dup_filter = None
        ddown_filter = None
        dbias = None
        dsign = None
        dsign_offset_x = None
        dsign_offset_y = None

        if ctx.needs_input_grad[0] or ctx.need_input_grad[1]:
            padding = [
                (up_filter.shape[-1] - 1) + (down_filter.shape[-1] - 1) - pad_x0,
                x_w * up - y_w * down + pad_x0 - (up - 1),
                (up_filter.shape[0] - 1) + (down_filter.shape[0] - 1) - pad_y0,
                x_h * up - y_h * down + pad_y0 - (up - 1),
            ]
            gain2 = gain * (up ** 2) / (down ** 2)
            sign_offset_x = sign_offset_x - (up_filter.shape[-1] - 1) + pad_x0
            sign_offset_y = sign_offset_y - (up_filter.shape[0] - 1) + pad_y0

            dx = FilteredLReLU.apply(
                dy,
                None,
                down_filter,
                up_filter,
                down,
                up,
                padding,
                negative_slope,
                gain2,
                None,
                not flip_filter,
                sign,
                sign_offset_x,
                sign_offset_y,
            )

        if ctx.needs_input_grad[1]:
            dbias = dx.sum((0, 2, 3))

        return (
            dx,
            dbias,
            dup_filter,
            ddown_filter,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dsign,
            dsign_offset_x,
            dsign_offset_y,
        )
