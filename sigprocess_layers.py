# -*- coding: utf-8 -*-
"""
@author: Longbiao Cheng
@modified: Chengzhong Wang
"""
import torch as torch
import torch.nn as nn
import numpy as np

class ShortTimeCepstrum(nn.Module):
    def __init__(self,fftsize,window_size,stride):
        super(ShortTimeCepstrum,self).__init__()
        self.fftsize = fftsize
        self.window_size = window_size
        self.stride = stride
        self.window_func = np.hamming(self.window_size)
        self.padd = self.window_size // 2
    def forward(self, input): # [B, 1, n_sample]
        # input = torch.nn.functional.pad(input,(self.padd,self.padd),mode='reflect')
        input = input.unfold(-1, self.window_size, self.stride) * torch.from_numpy(self.window_func).float().to(input.device)# [B, 1, T, W]
        spectrum = torch.fft.rfft(input, n = self.fftsize, dim=-1) # [B, 1, T, F]
        log_abs_spectrum = torch.log(torch.abs(spectrum) + 1e-20)# [B, 1, T, F]
        cepstrum = torch.fft.irfft(log_abs_spectrum, n = self.fftsize, dim=-1) # [B, 1, T, W]

        n_co = int(29)
        cepstrum_vocal_tract = torch.zeros_like(cepstrum)
        cepstrum_excitation = torch.zeros_like(cepstrum)
        for n in range(cepstrum.shape[-1]):
            mask_le_nco = 1 if (n < n_co or n >= 512-n_co)  else 0 # torch.tensor(abs(n-256)>=256-n_co).float().to(input.device)
            mask_gt_nco = 1 if (n >= n_co and n < 512-n_co) else 0# torch.tensor(abs(n-256)<256-n_co).float().to(input.device)
            mask_mp = 1 if n == 0  else 2 if n < 256 else 0
            cepstrum_vocal_tract[...,n] = cepstrum[...,n] * mask_le_nco * mask_mp # [B, 1, T, W]
            cepstrum_excitation[...,n] = cepstrum[...,n] * mask_gt_nco * mask_mp # [B, 1, T, W]
        vocal_tract = torch.exp(torch.real(torch.fft.rfft(cepstrum_vocal_tract, n = self.fftsize, dim=-1))).permute(0,1,3,2) # [B, 1, F, T]
        excitation = torch.exp(torch.real(torch.fft.rfft(cepstrum_excitation, n = self.fftsize, dim=-1))).permute(0,1,3,2) # [B, 1, F, T]
        return vocal_tract,excitation
    
class STFT(nn.Module):
    def __init__(self, fftsize, window_size, stride, trainable=False):
        super(STFT, self).__init__()
        self.fftsize = fftsize
        self.window_size = window_size
        self.stride = stride
        self.window_func = np.hamming(self.window_size)
        
        fcoef_r = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        fcoef_i = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        for w in range(self.fftsize//2+1):
            for t in range(self.window_size):
                fcoef_r[w, 0, t] = np.cos(2. * np.pi * w * t / self.fftsize)
                fcoef_i[w, 0, t] = -np.sin(2. * np.pi * w * t / self.fftsize)

        fcoef_r = fcoef_r * self.window_func
        fcoef_i = fcoef_i * self.window_func
        self.fcoef_r = torch.tensor(fcoef_r, dtype=torch.float)
        self.fcoef_i = torch.tensor(fcoef_i, dtype=torch.float)
        self.encoder_r = nn.Conv1d(1, self.fftsize//2+1, self.window_size, bias=False, stride=self.stride)
        self.encoder_i = nn.Conv1d(1, self.fftsize//2+1, self.window_size, bias=False, stride=self.stride)
        self.encoder_r.weight = torch.nn.Parameter(self.fcoef_r)
        self.encoder_i.weight = torch.nn.Parameter(self.fcoef_i)

        if trainable:
            self.encoder_r.weight.requires_grad = True
            self.encoder_i.weight.requires_grad = True
        else:
            self.encoder_r.weight.requires_grad = False
            self.encoder_i.weight.requires_grad = False

    def forward(self, input): # (B, 1, n_sample)

        spec_r = self.encoder_r(input)
        spec_i = self.encoder_i(input)
        output = torch.stack([spec_r, spec_i], dim=-1)
        output = output.permute([0, 2, 1, 3])

        return output # (B, T, F, 2)
class DivideWindow(nn.Module):
    def __init__(self,hop_size,window_size) -> None:
        super(DivideWindow,self).__init__()
        self.hop_size = hop_size
        self.window_size = window_size
        self.window_func = torch.from_numpy(np.hamming(self.window_size))
        self.padd = window_size // 2
    def forward(self,y):
        # y = torch.nn.functional.pad(y,(self.padd,self.padd),mode='reflect')
        output = y.unfold(-1, self.window_size,self.hop_size)
        output = output * self.window_func.to(y.device)
        output = output.permute(0, 1, 3, 2)
        return output # [B, 1, W, T]
    
class ISTFT(nn.Module):
    def __init__(self, fftsize, window_size, stride, trainable=False):
        super(ISTFT, self).__init__()
        self.fftsize = fftsize
        self.window_size = window_size
        self.stride = stride
        

        '''
        gain_ifft = (2.0*self.stride) / self.window_size
        self.window_func = gain_ifft * np.hanning(self.window_size)
        '''

        shiftSize = stride
        analysisWindow = np.hamming(self.window_size)
        synthesizedWindow = np.zeros(fftsize)
        for i in range(0, stride):
            amp = 0.0
            for j in range(0, fftsize // shiftSize):
                amp = amp + analysisWindow[i + j * shiftSize] * analysisWindow[i + j * shiftSize]
            for j in range(0, fftsize // shiftSize):
                synthesizedWindow[i + j * shiftSize] = analysisWindow[i + j * shiftSize] / amp
        self.window_func = synthesizedWindow
        
        coef_cos = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        coef_sin = np.zeros((self.fftsize//2 + 1, 1, self.window_size))
        for w in range(self.fftsize//2+1):
            alpha = 1.0 if w==0 or w==fftsize//2 else 2.0
            alpha /= fftsize
            for t in range(self.window_size):
                coef_cos[w, 0, t] = alpha * np.cos(2. * np.pi * w * t / self.fftsize)
                coef_sin[w, 0, t] = alpha * np.sin(2. * np.pi * w * t / self.fftsize)

        self.coef_cos = torch.tensor(coef_cos * self.window_func, dtype=torch.float)
        self.coef_sin = torch.tensor(coef_sin * self.window_func, dtype=torch.float)
        self.decoder_re = nn.ConvTranspose1d(self.fftsize//2+1, 1, self.window_size, bias=False, stride=self.stride)
        self.decoder_im = nn.ConvTranspose1d(self.fftsize//2+1, 1, self.window_size, bias=False, stride=self.stride)
        self.decoder_re.weight = torch.nn.Parameter(self.coef_cos)
        self.decoder_im.weight = torch.nn.Parameter(self.coef_sin)

        if trainable:
            self.decoder_re.weight.requires_grad = True
            self.decoder_im.weight.requires_grad = True
        else:
            self.decoder_re.weight.requires_grad = False
            self.decoder_im.weight.requires_grad = False

    def forward(self, input): # (B, T, F, 2)
        input = input.permute([0, 2, 1, 3]) # (B, F, T, 2)
        real_part = input[:, :, :, 0]
        imag_part = input[:, :, :, 1]

        time_cos = self.decoder_re(real_part)
        time_sin = self.decoder_im(imag_part)
        output = time_cos - time_sin

        return output  # (B, 1, n_sample)