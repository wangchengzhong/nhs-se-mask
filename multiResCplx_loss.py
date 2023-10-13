# -*- coding: utf-8 -*-
"""
@author: Cheng Longbiao
@modified: Chengzhong Wang
"""
import torch
import torch.nn as nn
from sigprocess_layers import STFT
import utils
import torch.nn.functional as F

class multiResCplx_mix_L1_loss(torch.nn.Module):
    def __init__(self):
        super(multiResCplx_mix_L1_loss, self).__init__()
        
        FilterLen_LST = [1200, 600, 240]
        self.stft_modules = nn.ModuleList()
        for filter_length in FilterLen_LST:
            self.stft_modules.append(STFT(fftsize=utils.next_pow_of_2(filter_length),window_size=filter_length, stride=filter_length//4, trainable=False))
        

    def forward(self, e_wav_i, t_wav_i, c_ratio=0.3):

        B, _, n_sample = t_wav_i.shape
        e_wav_i = e_wav_i[:, :, :n_sample]

        B, _, n_sample = e_wav_i.shape
        t_wav_i = t_wav_i[:, :, :n_sample]

        loss_sum = 0

        for batch in range(B):
            loss = 0
            t_wav = t_wav_i[batch].unsqueeze(0)
            e_wav = e_wav_i[batch].unsqueeze(0)
            for stft in self.stft_modules:
                targ_complex = stft(t_wav)
                enhc_complex = stft(e_wav)
                target_real, target_imag = targ_complex[..., 0], targ_complex[..., 1]
                target_mag, _ = self.realimag2magphase(target_real, target_imag)
                
                enhce_real, enhce_imag = enhc_complex[..., 0], enhc_complex[..., 1]
                enhce_mag, _ = self.realimag2magphase(enhce_real, enhce_imag)

                loss1 = torch.norm(enhce_mag - target_mag, p='fro') / (torch.norm(target_mag, p='fro')+1e-10)
                #loss2 = torch.norm((torch.log(enhce_mag+1e-10) - torch.log(target_mag+1e-10)), p=1) #/ (enhce_mag.shape[1] * enhce_mag.shape[2])
                loss2 = F.l1_loss(torch.log(enhce_mag), torch.log(target_mag))
                loss += loss1 + 0.5 * loss2
                # print(f'loss1: {loss1}, loss2: {loss2}')
            loss /= 3
                
            # print(f't_loss:{F.l1_loss(e_wav,t_wav)}')
            # loss += torch.norm(e_wav[i] - t_wav[i], p=1)
            loss += 50 * F.l1_loss(e_wav,t_wav)
            loss_sum += loss
        return loss_sum / B

    def realimag2magphase(self, real_part, imag_part):
        mag_part = torch.sqrt(torch.pow(real_part, 2) + torch.pow(imag_part, 2) + 1e-10)
        phase_part = torch.atan2(imag_part, real_part)

        return mag_part, phase_part
    
    def magphase2realimag(self, mag_part, phase_part):       
        real_part = mag_part * torch.cos(phase_part)
        imag_part = mag_part * torch.sin(phase_part)

        return real_part, imag_part
