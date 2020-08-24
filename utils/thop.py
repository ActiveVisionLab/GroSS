from thop import profile
from model.series_decomp import SeriesDecompConv
from torch.nn import Linear
import torch

def count_macs_conv(input_size, m, k, s, p, g, b):
    n, h, w = input_size
    # compute output size
    h_ = (h + 2 * p - k) // s + 1
    w_ = (w + 2 * p - k) // s + 1
    output_size = (m, h_, w_)

    macs = ((n * m * k * k) // g) * h_ * w_
    if b is not None:
        macs += m * h_ * w_
    return macs, output_size

def count_series_decomp_conv(layer, x, y):
    x = x[0]
    b, n, h, w = x.shape
    if isinstance(layer.bottle_dim, list):
        s, t = layer.bottle_dim
    else:
        s = layer.bottle_dim
        t = layer.bottle_dim
    out_channels = layer.out_channels
    kernel_size = layer.kernel_size
    stride = layer.stride[0]
    padding = layer.padding[0]
    num_groups = n // layer.test_gs
    macs_pw1, input_size = count_macs_conv(
        (n, h, w), s, 1, 1, 0, 1, None
    )
    macs_gc, input_size = count_macs_conv(
        input_size, t, kernel_size, stride, padding, num_groups, None
    )
    macs_pw2, input_size = count_macs_conv(
        input_size, out_channels, 1, 1, 0, 1, layer.bias
    )

    layer.total_ops += torch.DoubleTensor([b * sum([macs_pw1, macs_gc, macs_pw2])])

def count_macs(model, input_size):
    c, h, w = input_size
    inp = torch.randn(1, c, h, w).cuda()
    macs, _ = profile(model, inputs=(inp, ), custom_ops={SeriesDecompConv: count_series_decomp_conv}, verbose=False)
    model = model.cuda()
    return int(macs)