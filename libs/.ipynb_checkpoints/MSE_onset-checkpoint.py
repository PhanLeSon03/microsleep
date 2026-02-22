from tensorflow import keras
from libs.tf_model_base import TFModelBase
import scipy.io as spio

class MSEOnset(TFModelBase):    
    SAMPLING_RATE = 200
    HOP_SIZE = 32
    N_FFT = 128
    CHANNLES = 4
    N_CLASS = 1    
  
    N_FEATURE_FFT = N_FFT//2+1
    
    N_EEG = 21
    N_EOG = 8


    STEP_FRAME = 4
    PICKING_FRAME_SIZE = 32
    WIN_WAVE = 2*SAMPLING_RATE
    
    ONSET_TIME_THRESHOLD = STEP_FRAME * (HOP_SIZE/SAMPLING_RATE) 
    STEP_TIME = STEP_FRAME * (HOP_SIZE/SAMPLING_RATE) 
    PICKING_FRAME_TIME = PICKING_FRAME_SIZE * STEP_TIME
    PICKING_FRAME_STEP = int(PICKING_FRAME_SIZE/16)
    
        
    
    @classmethod
    def load_recording(cls, f_name, n_cl=4):
        mat = spio.loadmat(f_name, struct_as_record=False, squeeze_me=True)
        labels_O1 = mat['Data'].labels_O1
        labels_O2 = mat['Data'].labels_O2

        eeg_O1 = mat['Data'].eeg_O1
        eeg_O2 = mat['Data'].eeg_O2
        LOC = mat['Data'].E2
        ROC = mat['Data'].E1

   
        return (eeg_O1, eeg_O2, LOC, ROC, labels_O1, labels_O2)