# -*- coding: utf-8 -*-
"""
@author: Chengzhong Wang
"""
from dprnn import DPRNN
from cbam import CBAM 
from enc_decoder import Encoder,Decoder
import torch
import torch.nn as nn
import config as cf
import numpy as np
import torch.fft as fft
import torch.nn.functional as F
from merge import MergeBlock
from overlap_add import OverlapAdd
from sigprocess_layers import STFT,ShortTimeCepstrum,DivideWindow

def create_decoder_layers(last_channel_size):
    return [((128,32),(1,1),(1,1),(0,0)),
                  ((64,16),(5,2),(2,1),(1,1)),
                  ((32,last_channel_size),(5,2),(2,1),(1,1))]

encoder_layers = [((4,16), (5,2), (2,1), (1,1)),
                  ((16,32),(5,2),(2,1),(1,1)),
                  ((32,64),(1,1),(1,1),(0,0))]
class NHSMagMaskSE(nn.Module):
    def __init__(self):
        super(NHSMagMaskSE,self).__init__()
        # self.stft = STFT(fftsize=cf.fftsize,window_size=cf.window_length,stride=cf.hop_length)
        self.stcepstrum = ShortTimeCepstrum(fftsize=cf.fftsize,window_size=cf.window_length,stride=cf.hop_length)
        self.encoder_blocks = Encoder(encoder_layers)
        self.excitation_decoder = Decoder(create_decoder_layers(1),is_excitation=True)
        self.vocal_tract_decoder = Decoder(create_decoder_layers(1))
        self.complex_spectrum_decoder = Decoder(create_decoder_layers(2))
        rnn_size = cf.num_freq_bins//4 # (cf.batch_size,cf.num_freq_bins//4,cf.chunk_size) # [B, F, T]
        self.dprnns = nn.Sequential(*(DPRNN("LSTM",rnn_size,cf.hidden_size,rnn_size) for _ in range(2)))
        self.cbams = nn.ModuleList([nn.ModuleList([CBAM(16),CBAM(32),CBAM(64)]) for _ in range(3)])
        self.merge = MergeBlock()
        self.divide_window = DivideWindow(hop_size=cf.hop_length, window_size=cf.window_length)
        self.overlap = OverlapAdd(frame_length=cf.window_length, hop_size=cf.hop_length)
    def forward(self,y):
        if cf.debug: print(f'\n\n\nbegin -----input shape: {y.shape}\n')
        # y_in = self.stft(y) # [B, T, F, 2]
        y_d = self.divide_window(y)
        spectrum = torch.fft.rfft(y_d.permute(0,1,3,2), n = cf.fftsize, dim=-1) # [B, 1, T, W] -> [B, 1, T, F]
        y_r_c = torch.view_as_real(spectrum.squeeze(1)).permute(0,3,2,1).float() # [B, 1, T, F] -> [B, T, F, 2] -> [B, 2, F, T]
        # y_in = torch.stft(y.squeeze(1),n_fft=cf.fftsize,hop_length=cf.hop_length,win_length=cf.window_length,window=torch.hamming_window(cf.window_length).to(y.device),return_complex=False,center=False)
        # if cf.debug: print(f"\ny_in shape: {y_in.shape}")
        # y_r_c = y_in.permute(0,3,2,1) # [B, 2, F, T];  Y_r, Y_c # for 传统香烟
        # y_r_c = y_in.permute(0,3,1,2) # for 顶针
        if cf.debug: print(f'stft y_r_c shape: {y_r_c.shape}\n')
        vocal_tract, excitation = self.stcepstrum(y) # [B, 1, F, T]
        x = torch.cat([y_r_c, vocal_tract, excitation], dim=1) # [B, 4, F, T]
        if cf.debug: print(f'input x shape: {x.shape}')
        encoder_outputs = self.encoder_blocks(x)
        if cf.detail_debug: print(f'\nbefore encoder 1: {encoder_outputs[0].shape} before encoder 2:{encoder_outputs[1].shape} before encoder 3:{encoder_outputs[2].shape}after encoder 3:{encoder_outputs[3].shape}\n\n')
        medium = [[self.cbams[j][i](encoder_outputs[i]) for i in range(3)] for j in range(3)]
        o = self.dprnns(encoder_outputs[2])
        o_excitation = self.excitation_decoder(o,medium[0])# * excitation
        o_vocal_tract = self.vocal_tract_decoder(o,medium[1])# * vocal_tract
        o_complex_spectrum = self.complex_spectrum_decoder(o,medium[2])
        if cf.debug: print(f'\n------------After Decoder Shape:---------------\
                           \nexcitation: {o_excitation.shape} vocal_tract:{o_vocal_tract.shape} \
                           \ncomplex_spec:{o_complex_spectrum.shape} after dprnn shape:{o.shape}\n')
        x_nhs = self.synthesize_denoised_speech(o_excitation, o_vocal_tract, y_r_c)
        x_c = self.apply_mask_and_ifft(o_complex_spectrum, x[:,0:1,:,:], x[:,1:2,:,:])
        if cf.debug: print(f'\n-------------After Post-Processing Shape:--------------\
                           \nx_nhs: {x_nhs.shape}, x_c: {x_c.shape}, y_d: {y_d.shape}\n')
        output = self.merge(x_nhs, x_c, y_d)
        if cf.debug: print(f'\n--------------Mergeing shape------------\nmerge shape: {output.shape}\n')
        output = self.overlap(output)
        x_nhs_o = self.overlap(x_nhs.clone().detach())
        x_c_o = self.overlap(x_c.clone().detach())
        if cf.debug: print(f'------------final output shape: {output.shape}-------------\n')
        return output, x_nhs_o, x_c_o

    def synthesize_denoised_speech(self, o_e, o_h, y):
        # o_e_ifft = torch.fft.irfft(o_e, dim=2,n=512) # [B, 1, W, T]
        # o_h_ifft = torch.fft.irfft(o_h, dim=2,n=512) # [B, 1, W, T]
        # x_mnp = self.circular_convolution(o_e_ifft, o_h_ifft) #[B,1,W,T]
        x_mnp = torch.fft.irfft(o_e * o_h, dim=2,n=cf.fftsize)
        if cf.detail_debug: print(f'after ifft conv x_mnp shape:{x_mnp.shape}\n')
        x_mnp_fft = torch.fft.rfft(x_mnp, n=cf.fftsize, dim=2)
        x_mnp_fft_mag = torch.abs(x_mnp_fft)
        if cf.detail_debug: print(f'mnp fft mag shape: {x_mnp_fft_mag.shape}')
        y_permuted = y.permute(0,2,3,1)
        y_complex = torch.view_as_complex(y_permuted).unsqueeze(1)
        x_nhs = torch.fft.irfft(x_mnp_fft_mag * y_complex/(torch.abs(y_complex+1e-10)),n=cf.fftsize,dim=2)[:,:,0:cf.window_length,:]
        return x_nhs
    
    def apply_mask_and_ifft(self, o_c, Y_r, Y_i):
        Y = torch.cat([Y_r, Y_i], dim=1)
        masked_Y = o_c * Y
        masked_Y = masked_Y.permute(0,2,3,1) # [B, F, T, 2]
        if cf.detail_debug: print(f'masked_Y shape: {masked_Y.shape}\n')
        masked_Y_complex = torch.view_as_complex(masked_Y.contiguous()).unsqueeze(1) # [B, 1, F, T]
        x_c = torch.fft.irfft(masked_Y_complex, dim=2, n=cf.fftsize)[:,:,0:cf.window_length,:]
        if cf.detail_debug: print(f'x_c shape: {x_c.shape}\n')
        return x_c
    def circular_convolution(self, signal1, signal2):
        signal1 = signal1.permute(0,1,3,2).squeeze(1) # [B, 1, T, W]
        signal2 = signal2.permute(0,1,3,2).squeeze(1)
        signal_length = signal1.size(-1)
        result = torch.zeros_like(signal1)
        for i in range(signal_length):
            shifted_signal2 = torch.roll(signal2, shifts=-i, dims=-1)
            result[...,i] = torch.sum(signal1 * shifted_signal2, dim=-1)
        return result.unsqueeze(1).permute(0,1,3,2)