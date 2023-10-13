# -*- coding: utf-8 -*-
"""
@author: Chengzhong Wang
"""
import torch
import torch.nn as nn
class Conv2dBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,use_sigmoid=False):
        super(Conv2dBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        if use_sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.PReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)
    
class MergeBlock(nn.Module):
    def __init__(self) -> None:
        super(MergeBlock, self).__init__()
        # self.conv2d_block = nn.Sequential(
        #     nn.Conv2d(3,3,kernel_size=(5,3),stride=(1,1),padding=(2,1)),
        #     nn.Conv2d(3,3,kernel_size=(5,3),stride=(1,1),padding=(2,1)),
        #     nn.Conv2d(3,1,kernel_size=(5,3),stride=(1,1),padding=(2,1))
        # )
        self.conv2d_block1 = Conv2dBlock(3,3,(5,3),(1,1),(2,1))
        self.conv2d_block2 = Conv2dBlock(3,3,(5,3),(1,1),(2,1))
        self.conv2d_block3 = Conv2dBlock(3,1,(5,3),(1,1),(2,1),use_sigmoid=True)

    def forward(self, x_nhs, x_c, y):
        x_mb = torch.stack([x_nhs, x_c, y], dim=1).squeeze(2).float() # B * 3 * W * T
        x_mb = self.conv2d_block1(x_mb) # B * 1 * W * T
        x_mb = self.conv2d_block2(x_mb)
        D_m = self.conv2d_block3(x_mb)
        x_o = (1 - D_m) * x_nhs + D_m * x_c
        return x_o