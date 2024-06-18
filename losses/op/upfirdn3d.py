from collections import abc
import os

import torch
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
upfirdn3d_op = load(
    "upfirdn3d",
    sources=[
        os.path.join(module_path, "upfirdn3d.cpp"),
        os.path.join(module_path, "upfirdn3d_kernel.cu"),
    ],
)


class UpFirDn3dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y, up_z = up
        down_x, down_y, down_z = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1, g_pad_z0, g_pad_z1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], out_size[2], 1)

        grad_input = upfirdn3d_op.upfirdn3d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            down_z,
            up_x,
            up_y,
            up_z,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
            g_pad_z0,
            g_pad_z1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3], in_size[4])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.up_z = up_z
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.down_z = down_z
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.pad_z0 = pad_z0
        ctx.pad_z1 = pad_z1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], ctx.in_size[4], 1)

        gradgrad_out = upfirdn3d_op.upfirdn3d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.up_z,
            ctx.down_x,
            ctx.down_y,
            ctx.down_z,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
            ctx.pad_z0,
            ctx.pad_z1,
        )
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1], ctx.out_size[2]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None


class UpFirDn3d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y, up_z = up
        down_x, down_y, down_z = down
        pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1 = pad

        kernel_d, kernel_h, kernel_w = kernel.shape
        batch, channel, in_d, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_d, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1, 2]))

        out_d = (in_d * up_z + pad_z0 + pad_z1 - kernel_d + down_z) // down_z
        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_d, out_h, out_w)

        ctx.up = (up_x, up_y, up_z)
        ctx.down = (down_x, down_y, down_z)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_z0 = kernel_d - pad_z0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
        g_pad_z1 = in_d * up_z - out_d * down_z + pad_z0 - up_z + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1, g_pad_z0, g_pad_z1)

        out = upfirdn3d_op.upfirdn3d(
            input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1
        )
        out = out.view(-1, channel, out_d, out_h, out_w)

        return out

    @staticmethod
    def backward(ctx, grad_output):
        kernel, grad_kernel = ctx.saved_tensors

        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = UpFirDn3dBackward.apply(
                grad_output,
                kernel,
                grad_kernel,
                ctx.up,
                ctx.down,
                ctx.pad,
                ctx.g_pad,
                ctx.in_size,
                ctx.out_size,
            )

        return grad_input, None, None, None, None


def upfirdn3d(input, kernel, up=1, down=1, pad=(0, 0, 0)):
    if not isinstance(up, abc.Iterable):
        up = (up, up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down, down)

    if len(pad) == 3:
        pad = (pad[0], pad[1], pad[2], pad[0], pad[1], pad[2])

    if input.device.type == "cpu":
        out = upfirdn3d_native(input, kernel, *up, *down, *pad)

    else:
        out = UpFirDn3d.apply(input, kernel, up, down, pad)

    return out


def upfirdn3d_native(
    input, kernel, up_x, up_y, up_z, down_x, down_y, down_z, pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1
):
    _, channel, in_d, in_h, in_w = input.shape
    input = input.reshape(-1, in_d, in_h, in_w, 1)

    _, in_d, in_h, in_w, minor = input.shape
    kernel_d, kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_d, 1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0, 0, up_z - 1])
    out = out.view(-1, in_d * up_z, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), max(pad_z0, 0), max(pad_z1, 0)]
    )
    out = out[
        :,
        max(-pad_z0, 0) : out.shape[1] - max(-pad_z1, 0),
        max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[3] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 4, 1, 2, 3)
    out = out.reshape(
        [-1, 1, in_d * up_z + pad_z0 + pad_z1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1, 2]).view(1, 1, kernel_d, kernel_h, kernel_w)
    out = F.conv3d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_d * up_z + pad_z0 + pad_z1 - kernel_d + 1,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 4, 1)
    out = out[:, ::down_z, ::down_y, ::down_x, :]

    out_d = (in_d * up_z + pad_z0 + pad_z1 - kernel_d + down_z) // down_z
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_d, out_h, out_w)
