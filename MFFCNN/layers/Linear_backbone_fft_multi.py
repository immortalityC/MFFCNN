import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.RevIN import RevIN

import math







class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))


        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res








class patching(nn.Module):
    def __init__(self, patch, stride, seq_len, enc_in, batch_size,use_norm,dorpout):
        super(patching, self).__init__()
        self.conv = nn.Sequential(
            Inception_Block_V1(enc_in, enc_in*2 , num_kernels=3),
            nn.GELU(),
            Inception_Block_V1(enc_in*2 , enc_in, num_kernels=3)
        ).to('cuda')

        self.conv_ch = nn.Sequential(
            Inception_Block_V1(enc_in, enc_in*2 , num_kernels=3),
            nn.GELU(),
            Inception_Block_V1(enc_in*2 , enc_in, num_kernels=3)
        ).to('cuda')
        self.dropout = dorpout
        self.individual = False
        self.drop = nn.Dropout(dorpout)
        self.batch_size = batch_size
        self.enc_in = enc_in
        self.padding_patch = 'end'
        self.patch = patch
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.stride = stride
        self.flatten = nn.Flatten(start_dim=-2)
        self.relu = nn.GELU()
        self.seq_len = seq_len

        patch_num = int((self.seq_len - self.patch) / stride + 1)
        self.patch_num = patch_num
        self.use_norm =use_norm
        self.i_flatten = nn.Flatten(start_dim=-2)
        self.BatchNorm1d_2 = nn.BatchNorm2d(self.enc_in)
        self.i_linear = nn.Linear(2 * self.patch * (patch_num+1), self.seq_len)
        if self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

    def forward(self, x):
        org = x
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch, step=self.stride)
        org_data = x  # 将输入张量移到 CUDA 设备上
        x1 = x
        x1 = self.flatten(x1)
        x = self.conv(x)

        # x = self.drop(x)

        x_2 = x
        x = self.relu(x)



        x = x.permute(0, 2, 1, 3)
        in_channels = x.size(1)
        self.conv_ch = nn.Sequential(
            Inception_Block_V1(in_channels, in_channels, num_kernels=6),
            nn.GELU(),
            Inception_Block_V1(in_channels, in_channels, num_kernels=6)
        ).to('cuda')
        x = self.conv_ch(x)
        # x = self.drop(x)

        x = x.permute(0, 2, 1, 3)

        x = x + x_2
        x = self.relu(x)

        x = self.pool(x)

        x = self.flatten(x)
        L1 = nn.Linear(x.shape[2], x1.shape[2]).to('cuda')

        x = L1(x)
        x =self.drop(x)
        x = x.view(self.batch_size, self.enc_in, self.patch_num, self.patch)
        if self.use_norm:
            x = self.BatchNorm1d_2(x)

        # if self.individual:
        #     x_out = []
        #     for i in range(self.enc_in):
        #         z = x[:, i, :]
        #         o = org_data[:, i, :]
        #         o1 = org[:, i, :]
        #         z = self.flattens[i](z)
        #         o = self.flattens[i](o)
        #         z = torch.cat((z, o), dim=-1)
        #         z = self.linears[i](z)
        #         z = z + o1
        #         x_out.append(z)
        #     x = torch.stack(x_out, dim=1)
        #     x = x.permute(0, 2, 1)
        # else:
        x = self.i_flatten(x)
        org_data = self.i_flatten(org_data)
        x = torch.cat((x, org_data), dim=-1)
        x = self.i_linear(x)
        x = self.drop(x)
        # y = nn.Linear(x.shape[2], self.seq_len).to('cuda')
        # x = y(x)
        return x





class Model(nn.Module):
    """
    Just one Linear layer
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.dropout = configs.dropout

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.enc_in = configs.enc_in
        self.batch_size = configs.batch_size
        self.ll = configs.ll_num
        self.nl = configs.nl_num
        self.total_con = False
        self.duo_wei = False
        self.use_norm = configs.use_norm
        self.relu = nn.GELU()
        self.drop = nn.Dropout(self.dropout)
        self.drop_con = nn.Dropout(0.05)

        self.stride = configs.stride
        self.padding_patch = configs.padding_patch
        self.flatten = nn.Flatten(start_dim=-2)
        self.context_window = configs.seq_len
        self.individual = False
        self.is_con = False
        self.is_con1 = False
        self.is_norm = False
        self.revin = False

        if self.revin:
            pass

        self.norm = 'layer'


        patch_num = int((self.seq_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num

        if self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1

        else:
            self.i_flatten = nn.Flatten(start_dim=-2)
            self.i_linear = nn.Linear(2 * self.patch_len * self.patch_num, self.seq_len)
            self.i_linear_1 = nn.Linear(2 * self.seq_len, self.seq_len)
            self.i_dropout = nn.Dropout(0)

        self.t_conv1 = nn.Conv1d(self.enc_in, self.enc_in, kernel_size=3, stride=1, padding=1)

        self.conv1 = nn.Conv2d(self.enc_in, self.enc_in, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool = nn.MaxPool1d(2)

        self.chihua = False
        self.max_pool = nn.AvgPool1d(kernel_size=2, stride=2)




        self.i_linear_1 = nn.Linear(2 * self.seq_len, self.seq_len)
        self.h_L = False
        self.path = True

        self.Linear3 = nn.Linear(3*self.seq_len, self.seq_len)
        self.layer_norm = nn.LayerNorm(self.seq_len)


        self.patching1 = patching(7,7, self.seq_len, self.enc_in, self.batch_size,self.use_norm,self.dropout)
        self.patching2 = patching(42,42, self.seq_len, self.enc_in, self.batch_size,self.use_norm,self.dropout)
        self.patching3 = patching(112,112, self.seq_len, self.enc_in, self.batch_size,self.use_norm,self.dropout)
        # self.patching4 = patching(319, 319, self.seq_len, self.enc_in, self.batch_size,self.use_norm)
        # self.patching5 = patching(168, 168, self.seq_len, self.enc_in, self.batch_size, self.use_norm)

        self.convs = nn.ModuleList()

    def forward(self, x):




        if self.is_norm:
            x = x.permute(0, 2, 1)
            x = self.layernorm(x)
            x = x.permute(0, 2, 1)

        if self.path:
            x = x.permute(0, 2, 1).to('cuda')
            org = x
            x1 = x2 =x3 =x4 =x5=x
            # y = patching(28, 28, self.seq_len, self.enc_in, self.batch_size).to('cuda')
            # x = y(x)

            x1= self.patching1(x1)
            x11 = x1 +org


            x2= self.patching2(x2)
            x22 =x2+org


            #
            x3 = self.patching3(x3)
            x33 = x3+org



            #
            # x4 = self.patching4(x4)

            # x5 = self.patching5(x5)
            # x5 = org+x4

            x = torch.cat((x11,x22,x33),dim=2)
            x = self.Linear3(x)
            x = self.drop(x)

            if self.nl:
                x = self.relu(x)
                x = self.drop(x)
            x = org + x
            if self.use_norm:
                x = self.layer_norm(x)
            for i in range(self.ll):
                x = torch.cat((x, org), dim=-1)
                x = self.i_linear_1(x)
                x = org + x
                if self.use_norm:
                    x = self.layer_norm(x)

            for i in range(self.nl):
                x = torch.cat((x, org), dim=-1)
                x = self.i_linear_1(x)
                x = self.relu(x)
                x = self.drop(x)
                x = org + x
                if self.use_norm:
                    x = self.layer_norm(x)

            x = x.permute(0, 2, 1)


        if self.revin:
            x = self.revin_layer(x, 'denorm')
        return x

# 将模型和数据都移到 CUDA 设备上
# model = Model(configs).to('cuda')