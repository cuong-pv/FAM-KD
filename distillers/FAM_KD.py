import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
import torch.nn.init as init

import math

from ._base import Distiller

def feat_loss(fstudent, fteacher):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        loss_all = loss_all + loss
    return loss_all

class FAM_KD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(FAM_KD, self).__init__(student, teacher)
        self.shapes = cfg.FAM_KD.SHAPES
        self.out_shapes = cfg.FAM_KD.OUT_SHAPES
        self.in_shapes = cfg.FAM_KD.IN_SHAPES
        in_channels = cfg.FAM_KD.IN_CHANNELS
        in_channels_x = cfg.FAM_KD.IN_CHANNELS_X

        out_channels = cfg.FAM_KD.OUT_CHANNELS
        self.ce_loss_weight = cfg.FAM_KD.CE_WEIGHT
        self.famkd_loss_weight = cfg.FAM_KD.FAMKD_WEIGHT


        self.temperature = cfg.KD.TEMPERATURE
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

        self.warmup_epochs = cfg.FAM_KD.WARMUP_EPOCHS
        self.stu_preact = cfg.FAM_KD.STU_PREACT
        self.max_mid_channel = cfg.FAM_KD.MAX_MID_CHANNEL

        self.guide_layers = cfg.FAM_KD.GUIDE_LAYERS
        atfs = nn.ModuleList()
       # mid_channel = min(512, in_channels[-1])
        mid_channel = cfg.FAM_KD.MID_CHANNELS
       # in_channels = [32, 64, 64,1]

       ## KD loss
        
        for idx, in_channel in enumerate(in_channels):
          #  print(idx)
            atfs.append(
                CROSSATF(
                    in_channel,
                    in_channels_x[idx],
                    mid_channel[idx],
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                    self.shapes[::-1][idx],
                    self.in_shapes[::-1][idx], 
                    self.out_shapes[::-1][idx] 
                )
            )
        self.atfs = atfs[::-1]

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.atfs.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.atfs.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, features_student = self.student(image)
        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)
        # get features
        if self.stu_preact:
            x = features_student["preact_feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        else:
            x = features_student["feats"] + [
                features_student["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
            ]
        x = x[::-1]
        results = []
        out_features, res_features = self.atfs[0](x[0], out_shape=self.out_shapes[0])
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x[1:], self.atfs[1:], self.shapes[1:], self.out_shapes[1:]
        ):
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        features_teacher = features_teacher["feats"] + [
            features_teacher["pooled_feat"].unsqueeze(-1).unsqueeze(-1)
        ]
        features_teacher = [features_teacher[i] for i in self.guide_layers]


        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_fam_kd = (
            self.famkd_loss_weight
            * min(kwargs["epoch"] / self.warmup_epochs, 1.0)
            * feat_loss(results, features_teacher)
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_fam_kd
         #   "loss_kl": loss_kd
        }
        return logits_student, losses_dict


class CROSSATF(nn.Module):
    def __init__(self, in_channel, in_channel_x, mid_channel, out_channel, fuse, shape, in_shape, out_shape):
        super(CROSSATF, self).__init__()
        
        if in_channel != mid_channel:
            self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU()
        )
            nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        else:
            self.conv1 = None

        if in_channel_x != mid_channel:
            self.conv1_x = nn.Sequential(
                    nn.Conv2d(in_channel_x, mid_channel, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mid_channel),
                    nn.ReLU()
            )
            nn.init.kaiming_uniform_(self.conv1_x[0].weight, a=1)  # pyre-ignore
        else:
            self.conv1_x = None

        if fuse:
            self.att_conv = AttentionConv(mid_channel, mid_channel)
        else:
            self.att_conv = None
        self.conv2 = nn.Sequential(
                FAM_Module(mid_channel, out_channel, out_shape),
                nn.BatchNorm2d(out_channel),
                nn.ReLU()
            )

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, x_channel, h, w = x.shape
        # transform student features
        x_residual = x
        if self.conv1_x is not None:
            x = self.conv1_x(x)
            x_residual = x
        
        if self.att_conv is not None:
            # reduce channel dimension of residual features
            if self.conv1 is not None:
                y = self.conv1(y)
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            x = self.att_conv(x, y)
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x




# source https://github.com/leaderj1001/Stand-Alone-Self-Attention/blob/master/attention.py
class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=4, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
       # print(out_channels)
        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x, y):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(y)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class FAM_Module(nn.Module):
    def __init__(self, in_channels, out_channels, shapes):
        super(FAM_Module, self).__init__()

        """
        feat_s_shape, feat_t_shape
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapes = shapes
      #  print(self.shapes)
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
       # self.out_channels = feat_t_shape[1]
        self.scale = (1 / (self.in_channels * self.out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.shapes, self.shapes, dtype=torch.cfloat))
        self.w0 = nn.Conv2d(self.in_channels, self.out_channels, 1)

        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        if isinstance(x, tuple):
            x, cuton = x
        else:
            cuton = 0.1
        batchsize = x.shape[0]
        x_ft = torch.fft.fft2(x, norm="ortho")
      #  print(x_ft.shape)
        out_ft = self.compl_mul2d(x_ft, self.weights1)
        batch_fftshift = batch_fftshift2d(out_ft)

        # do the filter in here
        h, w = batch_fftshift.shape[2:4]  # height and width
        cy, cx = int(h / 2), int(w / 2)  # centerness
        rh, rw = int(cuton * cy), int(cuton * cx)  # filter_size
        # the value of center pixel is zero.
        batch_fftshift[:, :, cy - rh:cy + rh, cx - rw:cx + rw, :] = 0
        # test with batch shift
        out_ft = batch_ifftshift2d(batch_fftshift)
        out_ft = torch.view_as_complex(out_ft)
        #Return to physical space
        out = torch.fft.ifft2(out_ft, s=(x.size(-2), x.size(-1)),norm="ortho").real
        out2 = self.w0(x)
        return self.rate1 * out + self.rate2*out2
    
def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)
def batch_fftshift2d(x):
    real, imag = x.real, x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None)
            if i != axis else slice(0, n, None)
            for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None)
            if i != axis else slice(n, None, None)
            for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
