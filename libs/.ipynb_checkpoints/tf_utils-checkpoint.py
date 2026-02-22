import tensorflow as tf
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks, layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import regularizers
from tensorflow.python.tf2 import enable
import datetime
import tensorflow.keras.backend as kb
from sklearn.metrics import cohen_kappa_score


class TFUtils():    

    @staticmethod
    def get_log_dir():
        return "C:/temp/tensorboard/scalars/" + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    @staticmethod
    def load_model(checkpoint_dir, filter, custom_objects=None, compile=False) -> keras.Model:
        latest_checkpoint = TFUtils.get_latest_filepath(checkpoint_dir, filter)

        if latest_checkpoint is not None:
            print("Restoring from", latest_checkpoint)
            if(custom_objects is None):
                return keras.models.load_model(latest_checkpoint, compile=compile)
            else:
                return keras.models.load_model(latest_checkpoint, custom_objects=custom_objects, compile=compile)
        return None

    @staticmethod
    def get_latest_filepath(checkpoint_dir, filter):
        checkpoints = []
        for name in os.listdir(checkpoint_dir):
            if(filter in name):
                checkpoints.append(checkpoint_dir + name)

        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        else:
            latest_checkpoint = None
        return latest_checkpoint
        

    @staticmethod
    def get_default_callbacks(name_path_prefix,
                              stop_patience=20,
                              reduce_learn_rate_patience=10,
                              monitor="val_loss", learn_rate_factor=0.8, log_csv_path=None):
        callbacks = [keras.callbacks.ModelCheckpoint(
                    filepath=name_path_prefix + "{loss:.3f}" + '.h5',
                    save_best_only=True,        
                    monitor=monitor,
                    verbose=1
                    ),
                    keras.callbacks.EarlyStopping(monitor=monitor, patience=stop_patience),
                    keras.callbacks.ReduceLROnPlateau(monitor=monitor,
                                                      patience=reduce_learn_rate_patience,
                                                      factor=learn_rate_factor,
                                                      verbose=True)
                    ]
        if(log_csv_path is not None):
            history_logger=tf.keras.callbacks.CSVLogger(log_csv_path, separator="\t", append=True)
            callbacks.append(history_logger)
        return callbacks


    @staticmethod
    def create_tensorboard_callback(logdir=None, update_freq='batch', histogram_freq=1, write_images=True):        
        if(logdir is None):
            logdir = TFUtils.get_log_dir()

        callback = keras.callbacks.TensorBoard(log_dir=logdir,
                                               update_freq=update_freq,
                                               histogram_freq=histogram_freq,
                                               write_images=write_images)
        return callback

    @staticmethod
    def limit_GPU_usage():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print("Num GPUs Available: ", len(gpus))
        virtual = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        tf.config.experimental.set_virtual_device_configuration(gpus[0], virtual)

    @staticmethod
    def Crop(dim: int, start: int, end: int, **kwargs):
        """Crops (or slices) a Tensor on a given dimension from start to end
            example : to crop tensor x[:, :, 5:10]
        Args:
            dim (int): dimension to split
            start (int): start position
            end (int): end position

        Returns:
            tensor: sliced tensor
        """
        def func(x):
            dimension = dim
            if dimension == -1:
                dimension = len(x.shape) - 1
            if dimension == 0:
                return x[start:end]
            if dimension == 1:
                return x[:, start:end]
            if dimension == 2:
                return x[:, :, start:end]
            if dimension == 3:
                return x[:, :, :, start:end]
            if dimension == 4:
                return x[:, :, :, :, start:end]

        return Lambda(func, **kwargs)

    @staticmethod
    def euclidean_distance(vectors):        
        (featsA, featsB) = vectors

        # compute the sum of squared distances between the vectors
        sumSquared = kb.sum(kb.square(featsA - featsB), axis=1, keepdims=True)

        # return the euclidean distance between the vectors
        return kb.sqrt(kb.maximum(sumSquared, kb.epsilon()))


    @staticmethod
    def interleave(first: np.ndarray, second: np.ndarray):
        """Interleave two arrays with the same length or mismatch by 1 element

        Args:
            first (np.ndarray): first array
            second (np.ndarray): second array
        """
        out = np.empty((first.size + second.size,), dtype=first.dtype)
        out[0::2] = first
        out[1::2] = second
        return out
    
    @staticmethod
    def kappa_metric(y_true, y_pred, n_cl = 4):
        # computes Cohen kappa per class
        y =  np.array(y_true) 
        y_ = np.array(y_pred) 
        res = []
        for c in range(n_cl):
            res.append(cohen_kappa_score(y==c, y_==c))
        return np.array(res)