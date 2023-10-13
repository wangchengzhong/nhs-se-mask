# -*- coding: utf-8 -*-
"""
@author: Chengzhong Wang
"""
import torch.nn as nn
import torch
import numpy as np
class OverlapAdd(nn.Module):
    def __init__(self, frame_length, hop_size):
        super(OverlapAdd,self).__init__()
        self.frame_length = frame_length
        self.hop_size = hop_size
        
        analysisWindow = np.hamming(self.frame_length)
        synthesizedWindow = np.zeros(self.frame_length)
        for i in range(0, hop_size):
            amp = 0.0
            for j in range(0, frame_length // hop_size):
                amp = amp + analysisWindow[i + j * hop_size] * analysisWindow[i + j * hop_size]
            for j in range(0, frame_length // hop_size):
                synthesizedWindow[i + j * hop_size] = analysisWindow[i + j * hop_size] / amp
        self.window_func = torch.from_numpy(synthesizedWindow)
    def forward(self,input):
        window_func = self.window_func.to(input.device)
        B,_,W,T = input.shape
        output = torch.zeros((B,1,W + (T-1) * self.hop_size),device = input.device)
        # print(f'output length: {W + (T-1) * self.hop_size}')
        for t in range(T):
            start = t * self.hop_size
            end = start + self.frame_length
            output[:,:,start:end] += window_func * input[:,:,:,t]
        # pad = self.frame_length // 2
        # output = output[:,:,pad:-pad]
        return output