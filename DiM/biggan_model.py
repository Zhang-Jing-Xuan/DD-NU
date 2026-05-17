import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)

class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.bn = batchnorm_2d(out_features, eps=1e-4, momentum=0.1, affine=False)

        self.gain = snlinear(in_features=in_features, out_features=out_features, bias=False)
        self.bias = snlinear(in_features=in_features, out_features=out_features, bias=False)

    def forward(self, x, y):
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x)
        return out * gain + bias

def embedding(num_embeddings, embedding_dim):
    return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=bias),
                         eps=1e-6)

class GenBlock(nn.Module):
    def __init__(self):
        super(GenBlock, self).__init__()
        self.g_cond_mtd = "cBN"

        self.bn1 = ConditionalBatchNorm2d(148, 384)
        self.bn2 = ConditionalBatchNorm2d(148, 384)

        self.activation = nn.ReLU(inplace=True)
        self.conv2d0 = snconv2d(in_channels=384, out_channels=384, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = snconv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = snconv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)

    def forward(self, x, affine):
        x0 = x
        x = self.bn1(x, affine)
        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv2d1(x)

        x = self.bn2(x, affine)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode="nearest")
        x0 = self.conv2d0(x0)
        out = x + x0
        return out

class SelfAttention(nn.Module):
    """
    https://github.com/voletiv/self-attention-GAN-pytorch
    MIT License

    Copyright (c) 2019 Vikram Voleti

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self, in_channels, is_generator):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        if is_generator:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                                  stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                                stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                              stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                                 stride=1, padding=0, bias=False)
        else:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                                  stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1,
                                                stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1,
                                              stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1,
                                                 stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch // 8, h * w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch // 8, h * w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch // 2, h * w // 4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch // 2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        return x + self.sigma * attn_g

def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=True)

class Generator(nn.Module):
    def __init__(self, z_dim=80, apply_attn=True, attn_g_loc=[2]):
        super(Generator, self).__init__()

        self.z_dim = 80
        self.g_shared_dim = 128
        self.g_cond_mtd = "cBN"
        self.num_classes = 100
        self.mixed_precision = False
        self.in_dims = [96 * 4, 96 * 4, 96 * 4]
        self.out_dims = [96 * 4, 96 * 4, 96 * 4]
        self.bottom = 4
        self.num_blocks = len(self.in_dims)
        self.chunk_size = z_dim // (self.num_blocks + 1)
        self.affine_input_dim = self.chunk_size
        assert self.z_dim % (self.num_blocks + 1) == 0, "z_dim should be divided by the number of blocks"

        info_dim = 0

        self.linear0 = snlinear(in_features=self.chunk_size, out_features=self.in_dims[0]*self.bottom*self.bottom, bias=True)

        if self.g_cond_mtd != "W/O": # cBN
            self.affine_input_dim += self.g_shared_dim
            self.shared = embedding(num_embeddings=self.num_classes, embedding_dim=self.g_shared_dim)

        self.blocks = []
        for index in range(self.num_blocks):
            self.blocks += [[
                GenBlock()
            ]]

            if index + 1 in attn_g_loc and apply_attn: # [2], True
                self.blocks += [[SelfAttention(self.out_dims[index], is_generator=True)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = batchnorm_2d(in_features=self.out_dims[-1])
        self.activation = nn.ReLU(inplace=True)
        self.conv2d5 = snconv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # ops.init_weights(self.modules, g_init)

    def forward(self, z, label, shared_label=None, eval=False):
        affine_list = []

        zs = torch.split(z, self.chunk_size, 1)
        z = zs[0]
        if self.g_cond_mtd != "W/O": # cBN
            if shared_label is None: # None
                shared_label = self.shared(label)
            affine_list.append(shared_label)
        if len(affine_list) == 0: # 1
            affines = [item for item in zs[1:]]
        else:
            affines = [torch.cat(affine_list + [item], 1) for item in zs[1:]]

        act = self.linear0(z)
        act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
        counter = 0
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                if isinstance(block, SelfAttention):
                    act = block(act)
                else:
                    act = block(act, affines[counter])
                    counter += 1

        act = self.bn4(act)
        act = self.activation(act)
        act = self.conv2d5(act)
        out = self.tanh(act)
        return out

class DiscOptBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=192):
        super(DiscOptBlock, self).__init__()

        self.conv2d0 = snconv2d(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.conv2d1 = snconv2d(in_channels=3, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = snconv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)


        self.activation = nn.ReLU(inplace=True)
        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x
        x = self.conv2d1(x)
        
        x = self.activation(x)

        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        
        x0 = self.conv2d0(x0)
        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels=192, out_channels=192, downsample=True):
        super(DiscBlock, self).__init__()
        self.apply_d_sn = True
        self.downsample = downsample

        self.activation = nn.ReLU(inplace=True)

        self.ch_mismatch = False

        if self.ch_mismatch or downsample: # downsample=True
            self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            

        self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2d2 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)


        self.average_pooling = nn.AvgPool2d(2)

    def forward(self, x):
        x0 = x

        x = self.activation(x)
        x = self.conv2d1(x)


        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch: # True
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)
        out = x + x0
        return out

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim), eps=1e-6)

class Discriminator(nn.Module):
    def __init__(self, apply_d_sn=True, apply_attn=True, attn_d_loc=[1], d_embed_dim="N/A",
                 num_classes=100, d_init="ortho", d_depth="N/A"):
        super(Discriminator, self).__init__()

        self.d_cond_mtd = "PD"
        self.aux_cls_type = "W/O"
        self.normalize_d_embed = False
        self.num_classes = 100
        self.mixed_precision = False
        self.in_dims = [3] + [96 * 2, 96 * 2, 96 * 2]
        self.out_dims = [96 * 2, 96 * 2, 96 * 2, 96 * 2]
        down = [True, True, False, False]

        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[
                    DiscOptBlock()
                ]]
            else:
                self.blocks += [[
                    DiscBlock(in_channels=self.in_dims[index],
                              out_channels=self.out_dims[index],
                              downsample=down[index])
                ]]

            if index + 1 in attn_d_loc and apply_attn:
                self.blocks += [[SelfAttention(self.out_dims[index], is_generator=False)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.activation = nn.ReLU(inplace=True)


        self.linear1 = snlinear(in_features=self.out_dims[-1], out_features=1, bias=True)



        self.embedding = sn_embedding(num_classes, self.out_dims[-1])

        # ops.init_weights(self.modules, d_init)

    def forward(self, x, label, eval=False, adc_fake=False):
        embed, proxy, cls_output = None, None, None
        mi_embed, mi_proxy, mi_cls_output = None, None, None
        info_discrete_c_logits, info_conti_mu, info_conti_var = None, None, None
        h = x
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)
        bottom_h, bottom_w = h.shape[2], h.shape[3]
        h = self.activation(h)
        h = torch.sum(h, dim=[2, 3])

        # adversarial training
        adv_output = torch.squeeze(self.linear1(h))


        adv_output = adv_output + torch.sum(torch.mul(self.embedding(label), h), 1)


            
        return {
            "h": h,
            "adv_output": adv_output,
            "embed": embed,
            "proxy": proxy,
            "cls_output": cls_output,
            "label": label,
            "mi_embed": mi_embed,
            "mi_proxy": mi_proxy,
            "mi_cls_output": mi_cls_output,
            "info_discrete_c_logits": info_discrete_c_logits,
            "info_conti_mu": info_conti_mu,
            "info_conti_var": info_conti_var
        }
