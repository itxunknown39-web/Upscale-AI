import sys
import types
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers) if nn else None

class ResidualDenseBlock_5C(nn.Module if nn else object):
    """Residual Dense Block with 5 convolutions."""
    def __init__(self, nf=64, gc=32, bias=True):
        if not nn:
            return
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module if nn else object):
    """Residual in Residual Dense Block (RRDB)."""
    def __init__(self, nf, gc=32):
        if not nn:
            return
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module if nn else object):
    """Pure PyTorch RRDBNet Architecture for Real-ESRGAN."""
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        if not nn:
            return
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, nf=num_feat, gc=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

# Register complete basicsr shim modules (archs, utils, download_util)
def register_basicsr_shim():
    try:
        import torchvision.transforms.functional as F_trans
        sys.modules['torchvision.transforms.functional_tensor'] = F_trans
    except Exception:
        pass

    if "basicsr" not in sys.modules:
        basicsr_mod = types.ModuleType("basicsr")
        sys.modules["basicsr"] = basicsr_mod

    if "basicsr.archs" not in sys.modules:
        archs_mod = types.ModuleType("basicsr.archs")
        sys.modules["basicsr.archs"] = archs_mod

    if "basicsr.archs.rrdbnet_arch" not in sys.modules:
        rrdbnet_mod = types.ModuleType("basicsr.archs.rrdbnet_arch")
        rrdbnet_mod.RRDBNet = RRDBNet
        sys.modules["basicsr.archs.rrdbnet_arch"] = rrdbnet_mod

    if "basicsr.utils" not in sys.modules:
        utils_mod = types.ModuleType("basicsr.utils")
        sys.modules["basicsr.utils"] = utils_mod

    if "basicsr.utils.download_util" not in sys.modules:
        from scripts.model_manager import load_file_from_url
        download_mod = types.ModuleType("basicsr.utils.download_util")
        download_mod.load_file_from_url = load_file_from_url
        sys.modules["basicsr.utils.download_util"] = download_mod

register_basicsr_shim()
