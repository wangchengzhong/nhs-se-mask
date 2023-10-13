debug = False
detail_debug = False
train_mode = True
test_model_name = 'md_00epoch_01999batch_0.001000lr_0.6223trbatloss_0.6156valoss.pkl'

fftsize = 512
hop_length = 100
window_length = 400
num_freq_bins = 257

learning_rate = 1e-3
factor = 0.5
patience = 5
early_stop = 7


hidden_size = 128
sample_rate = 16000  # 16000 or 8000

EPSILON = 1e-8


'''
[train]
'''
CLEAN_DATA_FOLDERS = ['C:/Users/wcz53/Documents/1_ioa/nhs_se_mask/dtb/mandarin_speech']#['/data/DataSets/clean_speech_with_vad', '/data/DataSets/challenge/DNS_2022/DNS-Challenge/datasets/clean']
NOISE_DATA_FOLDERS = ['C:/Users/wcz53/Documents/1_ioa/nhs_se_mask/dtb/noise']#['/data/DataSets/challenge/DNS_ICASSP2021/datasets/noise']
TSET_DATA_FOLDERS = 'C:/Users/wcz53/Documents/1_ioa/nhs_se_mask/dtb/VBD'

audio_length = 10
silence_length = 0.5
snr_lower = -5
snr_upper = 20
snr_improvement = 40
target_level_lower = -35
target_level_upper = -1

modulation_noise_ratio = 0.25

highpass_ratio = 0.15
hp_freq_min = 200
hp_freq_max = 500

lowpass_ratio = 0.15
lp_freq_min = 3500
lp_freq_max = sample_rate//2-250

target_type = 'sub_band'

# Configuration for training process
MODEL_NAME = 'NHSMagMaskSE'
MODEL_STORE_DIR = str(sample_rate) + '/'+ MODEL_NAME + '-' + str(snr_improvement)
CKPT_NAME= '' # model_name(*.pkl) or ''

EPOCH = 50
TRAIN_BATCH_SIZE = 3#36*3
VALIDATION_BATCH_SIZE = 1# 8*2
keep_model_batch_len = 2000
