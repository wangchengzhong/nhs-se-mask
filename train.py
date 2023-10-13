# -*- coding: utf-8 -*-
"""
@author: Cheng Longbiao
@modified: Chengzhong Wang
"""
import os
import time
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from utils import view_bar
from config import *
from multiResCplx_loss import multiResCplx_mix_L1_loss

from dataloader.offline_dataset import offlineDataset
from dataloader.online_dataset import onlineDataset
from dataloader.paired_dataloader import PairedDataLoader

from nhs_magse_mask import NHSMagMaskSE
import soundfile as sf

from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.multiprocessing as mp
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.cuda.manual_seed_all(123)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter+=1
            print(f'EarlyStopping counter:{self.counter}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model)
            self.counter = 0
    def save_checkpoint(self,val_loss,model):
        if self.verbose:
            print(f'validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saveing Model ...')
        torch.save(model.state_dict(), f'module_store/checkpoint_vl_{val_loss}.pt')
        self.val_loss_min = val_loss


def validation(net, criterion, validation_data_loader, val_batch_num):
    net.eval()
    val_loss = 0
    k = 1
    for val_batch_idx, val_batch_info in enumerate(validation_data_loader.get_data_loader()):
        inp = val_batch_info.noisy_wav_data.to(device='cuda')
        target = val_batch_info.clean_wav_data.to(device='cuda')[:,:,window_length:-window_length]
        
        wav_out,x_nhs,x_c = net(inp)
        wav_out = wav_out[:,:,window_length:-window_length]
        x_nhs = x_nhs[:,:,window_length:-window_length]
        x_c = x_c[:,:,window_length:-window_length]
        for i in range(wav_out.shape[0]):
            sf.write(f'validation_output/wav_out/wav_out_{k}.wav',wav_out[i,0,:].cpu().detach().numpy(),sample_rate)
            sf.write(f'validation_output/x_nhs/x_nhs_{k}.wav',x_nhs[i,0,:].cpu().detach().numpy(),sample_rate)
            sf.write(f'validation_output/x_c/x_c_{k}.wav',x_c[i,0,:].cpu().detach().numpy(),sample_rate)
            sf.write(f'validation_output/target/target_{k}.wav',target[i,0,:].cpu().detach().numpy(),sample_rate)
            sf.write(f'validation_output/noisy/inp_{k}.wav',inp[i,0,:].cpu().detach().numpy(),sample_rate)
            k+=1

        loss = criterion(wav_out, target)
        val_loss += loss.item() 

        view_bar('Validation', val_batch_idx+1, val_batch_num, val_loss/(val_batch_idx + 1), val_loss/(val_batch_idx + 1))

        del loss, inp, target
        torch.cuda.empty_cache()
    return val_loss / (val_batch_idx+1)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

def train(net, train_data_loader, validation_data_loader, epoch, model_dir, tr_batch_num, val_batch_num):

    criterion =  multiResCplx_mix_L1_loss().to(device='cuda').eval()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=7,verbose=True)
    tr_batloss, tr_win_loss = 0, 0
    step_num = 0
    
    best_model_name = "1"
    init_merge_stage = True
    for i in range(epoch):

        tr_batloss = 0
        start_time = time.time()
        tr_loss = 0

        net.train()            
        for batch_idx, batch_info in enumerate(train_data_loader.get_data_loader()):      

            step_num += 1

            inp = batch_info.noisy_wav_data.to(device='cuda')
            target = batch_info.clean_wav_data.to(device='cuda')

            optimizer.zero_grad()

            out_wav, x_nhs, x_c = net(inp) #, target
            B, _, n_sample = target.shape
            out_wav = out_wav[:, :, window_length:-window_length]

            B, _, n_sample = out_wav.shape
            target = target[:, :, window_length:-window_length]
            inp = inp[:, :, window_length:-window_length]
            if debug: print(f'out_wav shape: {out_wav.shape}, target shape: {target.shape}')

            loss = criterion(out_wav, target)

            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)
            
            loss.backward()
            optimizer.step()
            
            tr_loss += loss.item()
            tr_batloss += loss.item()
            tr_win_loss += loss.item()
            
            view_bar('Training  ', batch_idx+1, tr_batch_num, tr_batloss/(batch_idx%keep_model_batch_len+1), tr_loss /(batch_idx+1))

            del loss, inp, target, out_wav#, loss_e, loss_t
            torch.cuda.empty_cache()
            # validation and save model after some training steps 
            if (batch_idx+1)%keep_model_batch_len == 0:
                print("   \t")
                val_loss = validation(net, criterion, validation_data_loader, val_batch_num)
                scheduler.step(val_loss)
                early_stopping(val_loss, net)
                if early_stopping.early_stop:
                    print("Early Stopping!")
                    return
                else:
                    best_model_name = "md_%02depoch_%05dbatch_%.6flr_%.4ftrbatloss_%.4fvaloss.pkl" % (i, batch_idx, learning_rate, tr_batloss/keep_model_batch_len, val_loss)
                    print("   \t", best_model_name, "time used", time.time()-start_time)
                    torch.save(net, os.path.join(model_dir, best_model_name))

                    tr_batloss = 0
                    net.train()

        '''''' 
        # validation and save model after one epoch training
        val_loss = validation(net, criterion, validation_data_loader, val_batch_num)
        scheduler.step(val_loss)
        early_stopping(val_loss,net)
        if early_stopping.early_stop:
            print("Early Stopping!")
            return
        model_name = "md_%02depoch_%05dbatch_%.6flr_%.4ftrloss_%.4fvaloss.pkl" % (i, batch_idx, learning_rate, tr_loss/(batch_idx+1), val_loss)
        print('   \t', model_name, "time used", time.time()-start_time)
        torch.save(net, os.path.join(model_dir, model_name))
        #print("model in %02d epoch has been saved!" % (i))
              
        
if __name__ == '__main__':
    workspace = './'
    mp.set_start_method('spawn')
    if not os.path.exists(workspace):
        os.mkdir(workspace)
    # model
    MODEL_STORE = os.path.join(workspace, 'module_store')
    VALIDATION_STORE = os.path.join(workspace, 'validation_output')
    VALIDATION_STORE_SET = ['wav_out', 'x_nhs', 'x_c', 'target', 'noisy']
    for i in VALIDATION_STORE_SET:
        v_dir = os.path.join(VALIDATION_STORE,i)
        if not os.path.exists(v_dir):
            os.makedirs(v_dir)
    if not os.path.exists(MODEL_STORE):
        os.makedirs(MODEL_STORE)
        print('Create model store file successful!\n'
            'Path: \"{}\"'.format(MODEL_STORE))
    else:
        print('The model store path: {}'.format(MODEL_STORE))

    model_dir = os.path.join(MODEL_STORE, MODEL_STORE_DIR)
    
    if not os.path.exists(model_dir):
        #os.mkdirs(model_dir)
        os.makedirs(model_dir)
        print('Create model store file successful!\n'
                'Path: \"{}\"'.format(model_dir))
    else:
        print('The model store path: {}'.format(model_dir))
    if train_mode:
        print('reading validation data..........')
        validation_data_set = offlineDataset(TSET_DATA_FOLDERS, fs=sample_rate)
        validation_data_loader = PairedDataLoader(data_set=validation_data_set, batch_size=VALIDATION_BATCH_SIZE, is_shuffle=False)

        print('synthesize training data..........')
        train_data_set = onlineDataset(CLEAN_DATA_FOLDERS, NOISE_DATA_FOLDERS)
        train_data_loader = PairedDataLoader(data_set=train_data_set, batch_size=TRAIN_BATCH_SIZE, is_shuffle=True)
        
        tr_batch_num = train_data_set.__len__() // TRAIN_BATCH_SIZE
        val_batch_num = validation_data_set.__len__() // VALIDATION_BATCH_SIZE
        print("train batch numbers: %d, validation batch numbers: %d" % (tr_batch_num, val_batch_num))

        model = NHSMagMaskSE()
        model.apply(weights_init)  
        model = torch.nn.DataParallel(model)  # use multiple GPU 
        model = model.to(device='cuda')


        if CKPT_NAME != '':  
            model_ckpt = torch.load(os.path.join(model_dir, CKPT_NAME))
            model.load_state_dict(model_ckpt.state_dict()) 
            print('Loading pre-trained model from: ', os.path.join(model_dir, CKPT_NAME))
            
        total_params = sum(p.numel() for p in model.parameters())
        print('total parameters number:', total_params)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('total trainable parameters number:', total_trainable_params/1000000)
        
        train(model, train_data_loader, validation_data_loader, EPOCH, model_dir, tr_batch_num, val_batch_num)
    else:
        model = NHSMagMaskSE()
        model = (torch.load(os.path.join(model_dir, test_model_name),map_location='cuda'))
        model = model.to(device='cuda')
        model.eval()
        test_data_set = offlineDataset(TSET_DATA_FOLDERS, fs=sample_rate)
        test_data_loader = PairedDataLoader(data_set=test_data_set, batch_size=VALIDATION_BATCH_SIZE, is_shuffle=False)
        criterion =  multiResCplx_mix_L1_loss().to(device='cuda').eval()
        test_batch_num = test_data_set.__len__() // VALIDATION_BATCH_SIZE
        validation(model, criterion, test_data_loader, test_batch_num)