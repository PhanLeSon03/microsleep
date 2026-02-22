import os
from time import time
import librosa
import numpy as np
from tensorflow.python.data.ops.readers import TFRecordDataset
from libs.tf_data_record import TFMultiOutParserMIMO
import tensorflow as tf
from libs.MSE_onset import MSEOnset
from libs.misc import Misc
import librosa
import pathlib
import scipy.io as spio
from collections import Counter
from libs.signal_processing import eeg_features, eog_features

TWO_INPUT = True # setting one or two feature
WAVE_IN = False  # second feature: waveform or hand-crafted features

if TWO_INPUT:
    if MSEOnset.N_CLASS ==4:
        out_train_folder = './content/MSE_Onset_Database_2In_4Class/train/'
        out_validation_folder = './content/MSE_Onset_Database_2In_4Class/validation/'
    else:
        if WAVE_IN:
            out_train_folder = './content/MSE_Onset_Database_W/train/'
            out_validation_folder = './content/MSE_Onset_Database_W/validation/'
        else:
            out_train_folder = './content/MSE_Onset_Database_2In/train/'
            out_validation_folder = './content/MSE_Onset_Database_2In/validation/'
else:
        out_train_folder = './content/MSE_Onset_Database/train/'
        out_validation_folder = './content/MSE_Onset_Database/validation/'

        


data_folder_list = [
                    './../../data/files/',
                    ]

files_train = []
files_val = []
files_test = []
f_set = './../../data/file_sets.mat'
mat = spio.loadmat(f_set)
tmp = mat['files_train']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_train.extend(file)
tmp = mat['files_val']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_val.extend(file)
tmp = mat['files_test']
for i in range(len(tmp)):
    file = [str(''.join(l)) for la in tmp[i] for l in la]
    files_test.extend(file)
    
stats_path = f"./model_data/stats_{MSEOnset.CHANNLES}channels.mat"
stats = spio.loadmat(stats_path)

mean_STFT = stats["mean_STFT"].squeeze()
std_STFT  = stats["std_STFT"].squeeze()

print("Mean per channel:", mean_STFT)
print("Std per channel :", std_STFT)


SAMPLING_RATE = MSEOnset.SAMPLING_RATE
STEP_FRAME = MSEOnset.STEP_FRAME
HOP_SIZE = MSEOnset.HOP_SIZE
STEP_TIME = STEP_FRAME * (HOP_SIZE/SAMPLING_RATE)
PICKING_FRAME_SIZE = MSEOnset.PICKING_FRAME_SIZE       


def label_feature_generator(labels: np.ndarray, features: np.ndarray):    

    for class_labels, f_frame in zip(labels, features):
        if MSEOnset.N_CLASS == 1:
            class_labels[class_labels > 1] = 0                     
        
        y_dict = {'y1': class_labels}
        yield f_frame, y_dict
        
def label_2feature_generator(labels: np.ndarray, features: np.ndarray, features_1: np.ndarray):    

    for class_labels, f_frame, frame_1 in zip(labels, features, features_1):
        if MSEOnset.N_CLASS == 1:
            class_labels[class_labels > 1] = 0
                            
        y_dict = {'y1': class_labels}
        frame = [f_frame, frame_1]
        yield frame, y_dict

def stft_log_features(x, n_fft, hop_length, mean=0.0, var=1.0):
    STFT = librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length)                   # (freq, frames)
    log_compress = librosa.core.power_to_db(np.abs(STFT) ** 2)                  # (freq, frames)
    feat = log_compress.T                                                       # (frames, freq)
    feat = (feat - mean) / var
    return feat



def create_tfrecord_feature(data_folder, list_files, out_folder):    
    count = 0
    if(not os.path.exists(out_folder)):
        os.makedirs(out_folder)
    
    exist_files = os.listdir(out_folder)
    
    for file_name in list_files:                                    
        data_path = Misc.try_get_data(data_folder, file_name)
        if(data_path is None):  
            print(f'{file_name} not existed')
            continue
     
 
        
        eeg_O1, eeg_O2, LOC, ROC, labels_O1, labels_O2 = MSEOnset.load_recording(data_path)
        

        end_time_label = len(labels_O1)/SAMPLING_RATE
        print(f'{file_name} (second): {end_time_label}')
        eindex = len(labels_O1) 
        
        
        signals = [eeg_O1[:eindex], eeg_O2[:eindex], LOC[:eindex], ROC[:eindex]]
        
        feat_list = []
        for iC, ch_sig in enumerate(signals):
            feat_ch = stft_log_features(
                ch_sig,
                n_fft=MSEOnset.N_FFT,
                hop_length=HOP_SIZE,
                mean=mean_STFT[iC],
                var=std_STFT[iC]   
            )
            feat_list.append(feat_ch)
        
        features = np.stack(feat_list, axis=-1)  
        
        
        # Second input
        if WAVE_IN:
            wave_channels = len(signals)   # 4 (O1, O2, LOC, ROC)            
        else:
            hop = HOP_SIZE*STEP_FRAME
            window_size = hop

            EEG_feature = []
            EEG_features = []
            for iC, ch_sig in enumerate(signals[0:2]):
                num_steps = (len(ch_sig) - window_size + hop)//hop
                for i in range(num_steps):
                    start = i * hop
                    end = start + window_size
                    window_df = ch_sig[start:end]
                    welch_ = eeg_features(window_df, fs=SAMPLING_RATE)
                    EEG_feature.append(welch_) 
                EEG_features.append(EEG_feature)          
            EEG_features = np.concatenate(EEG_features, axis=1) 

            EOG_feature = []
            EOG_features = []
            for iC, ch_sig in enumerate(signals[2:]):
                num_steps = (len(ch_sig) - window_size + hop)//hop
                for i in range(num_steps):
                    start = i * hop
                    end = start + window_size
                    window_df = ch_sig[start:end]
                    welch_ = eog_features(window_df, fs=SAMPLING_RATE)
                    EOG_feature.append(welch_) 
                EOG_features.append(EOG_feature)          
            EOG_features = np.concatenate(EOG_features, axis=1) 


            Merge_features = np.concatenate([EEG_features, EOG_features], axis=1) 

        
        # picking frame
        frame_length = features.shape[0]
        freq_bins = features.shape[1]
        channels = features.shape[2]
        
     
        
        n_win_frames = int(np.ceil(frame_length / STEP_FRAME / MSEOnset.PICKING_FRAME_STEP))

        pick_STFT = np.zeros((n_win_frames,
                                  PICKING_FRAME_SIZE*STEP_FRAME,
                                  freq_bins,
                                  channels))
        
        win_len = MSEOnset.WIN_WAVE 
        if WAVE_IN:
            pick_wave = np.zeros((
                n_win_frames,
                PICKING_FRAME_SIZE,
                win_len,
                channels
            ))
            
        else:
            pick_EEG = np.zeros((n_win_frames,
                                      PICKING_FRAME_SIZE,
                                  2*MSEOnset.N_EEG + 2*MSEOnset.N_EOG))# second feature
        
        if MSEOnset.N_CLASS > 1:
            pick_labels = np.zeros((n_win_frames,PICKING_FRAME_SIZE, MSEOnset.N_CLASS)) # one-hot decode
        else:
            pick_labels = np.zeros((n_win_frames,PICKING_FRAME_SIZE))
        
        print('Number of frame ~20.5s: {}'.format(n_win_frames))

        n_frames = features.shape[0]
        for i, j in enumerate(range(0, n_frames, MSEOnset.PICKING_FRAME_STEP*STEP_FRAME)):   # 75% frame overlap      
            end_idx = j + PICKING_FRAME_SIZE * STEP_FRAME
            end_idx = end_idx if end_idx < n_frames else n_frames
            copy_size = end_idx - j
            pick_STFT[i, :copy_size, :,:] = features[range(j, end_idx), :,:]
            
            n_step = copy_size // STEP_FRAME 
            for iFrame in range(n_step):
                b = (j + iFrame * STEP_FRAME)*HOP_SIZE
                e = b + STEP_FRAME*HOP_SIZE
                block = labels_O1[b:e]
                if len(block) == 0:
                    represent = 0
                else:
                    represent = Counter(block).most_common(1)[0][0]

                if MSEOnset.N_CLASS > 1:
                    pick_labels[i, iFrame, represent] = 1
                else:
                    pick_labels[i, iFrame] = represent
             
            # second feature
            if WAVE_IN:
                for iFrame in range(n_step):
                    b = (j + iFrame * STEP_FRAME) * HOP_SIZE
                    e = b + win_len
                    
                    for ch_idx, ch_sig in enumerate(signals):
                        sig_len = len(ch_sig)
                        e = e if e < sig_len else sig_len
                        
                        #normalize
                        # mu = np.mean(ch_sig[b:e])
                        # sigma = np.std(ch_sig[b:e]) + 1e-6                        
                        pick_wave[i, iFrame, :e-b, ch_idx] = ch_sig[b:e]
     
            else:
                pick_EEG[i,:copy_size//STEP_FRAME]= Merge_features[j//STEP_FRAME: end_idx//STEP_FRAME]
            
            
        label_name = file_name.split('.')[0]
        out_file_path = os.path.join(out_folder,label_name) 
        normalized_out_file_path = out_file_path.encode('utf-8', errors='ignore')
        
        if TWO_INPUT:
            if WAVE_IN:
                generator = label_2feature_generator(pick_labels, pick_STFT, pick_wave)
            else:
                generator = label_2feature_generator(pick_labels, pick_STFT, pick_EEG)
            TFMultiOutParserMIMO.write_tfrecord_2inputs(normalized_out_file_path, generator)
        else:        
            generator = label_feature_generator(pick_labels, pick_STFT)  
            TFMultiOutParserMIMO.write_tfrecord_1input(normalized_out_file_path, generator)


        count += 1        

    print(f'{data_folder} - Total files: {count}')


for data_folder in data_folder_list:
    create_tfrecord_feature(data_folder, files_train, out_train_folder)
    create_tfrecord_feature(data_folder, files_val, out_validation_folder)    
    
    

# Review the generated tensorflow buffer
    
# test_file = os.listdir(out_train_folder)[0]
# print(f'test_file: {test_file}')
# test_path = os.path.join(out_train_folder, test_file)

# # Get written dataset
# x_shape = [MSEOnset.PICKING_FRAME_SIZE*MSEOnset.STEP_FRAME, MSEOnset.N_FEATURE_FFT, MSEOnset.CHANNLES]
# y1_shape = [MSEOnset.PICKING_FRAME_SIZE]

# y_shape = {'y1': y1_shape}
# data_sets = tf.data.TFRecordDataset(test_path, compression_type="GZIP")
# data_sets = data_sets.map(TFMultiOutParserMIMO.get_1out_parser(x_shape, y_shape))

# index = 0
# for j, (feature, label) in enumerate(data_sets):    
#     print(feature.shape)    
#     print(label.shape)
#     onset = label.numpy()
 
#     for i, o in enumerate(onset):
#         time = i * MSEOnset.STEP_TIME + j * MSEOnset.PICKING_FRAME_TIME / 2
#         print(f'Time: {time}\t{o}')     



# def compute_dataset_stats_per_channel(data_folder, list_files):
#     print("Computing dataset mean/std per channel from training set...")

#     n_channels = MSEOnset.CHANNLES
#     total_sum = np.zeros(n_channels)
#     total_sq_sum = np.zeros(n_channels)
#     total_count = np.zeros(n_channels)

#     for file_name in list_files:
#         data_path = Misc.try_get_data(data_folder, file_name)
#         if data_path is None:
#             continue

#         eeg_O1, eeg_O2, LOC, ROC, labels_O1, labels_O2 = MSEOnset.load_recording(data_path)

#         signals = [eeg_O1, eeg_O2, LOC, ROC]

#         for ch_idx, ch_sig in enumerate(signals):
#             feat = stft_log_features(
#                 ch_sig,
#                 n_fft=MSEOnset.N_FFT,
#                 hop_length=HOP_SIZE,
#                 mean=0.0,
#                 var=1.0
#             )

#             total_sum[ch_idx] += np.sum(feat)
#             total_sq_sum[ch_idx] += np.sum(feat ** 2)
#             total_count[ch_idx] += feat.size

#     mean = total_sum / total_count
#     var = (total_sq_sum / total_count) - (mean ** 2)
#     std = np.sqrt(var)

#     print("Channel means:", mean)
#     print("Channel stds :", std)

#     return mean, std

# mean_STFT, std_STFT = compute_dataset_stats_per_channel(data_folder_list[0], files_train)

# os.makedirs(os.path.dirname(stats_path), exist_ok=True)

# spio.savemat(stats_path, {
#     "mean_STFT": mean_STFT,
#     "std_STFT": std_STFT
# })

# print(f"Saved stats to: {stats_path}")

