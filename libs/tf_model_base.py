import abc
from typing import Tuple, List
from libs.misc import Misc
from libs.tf_utils import TFUtils
import pathlib
import tensorflow as tf
from tensorflow import keras
import numpy as np


class TFModelBase(metaclass=abc.ABCMeta):    

    MODEL_PATH = f'{pathlib.Path(__file__).parent.absolute()}/../model_data/'
    _IS_FORCE_CPU_GLOBALLY = False

    _MAX_GPU_FEATURE_SIZE = 25000000
    _MAX_CPU_FEATURE_SIZE = 10000000 

    @classmethod
    def version(self): 
        return "1.0"

    def __init__(self, model_name, is_force_cpu=True, custom_object=None, model=None, Model_TFlite = False, model_sub_name=None):
        self.is_force_cpu = is_force_cpu

        if(custom_object is None):
            custom_object = {"keras": keras}

        self.TFlite = Model_TFlite
        if Model_TFlite == True:
            if(model is not None):
                self.model = model
            else:
                model_path = TFModelBase.MODEL_PATH
                self.model = tf.lite.Interpreter(model_path=model_path)
                self.model.allocate_tensors()
            self.input_detail = self.model.get_input_details()[0]
            self.Onset = self.model.get_output_details()[0]
            self.Duration = self.model.get_output_details()[2]
            self.Pitch = self.model.get_output_details()[1]
        else:
            if(model is not None):
                self.model = model
            else:
                model_path = TFModelBase.MODEL_PATH

                if(is_force_cpu):
                    with tf.device('/CPU'):
                        self.model = TFUtils.load_model(model_path, model_name, custom_object)
                else:
                    self.model = TFUtils.load_model(model_path, model_name, custom_object)

            if model_sub_name is not None:
                model_path = TFModelBase.MODEL_PATH

                if (is_force_cpu):
                    with tf.device('/CPU'):
                        self.model_sub = TFUtils.load_model(model_path, model_sub_name, custom_object)
                else:
                    self.model_sub = TFUtils.load_model(model_path, model_sub_name, custom_object)
            else:
                self.model_sub = self.model


    @staticmethod  
    def foce_CPU_globally():
        if(TFModelBase._IS_FORCE_CPU_GLOBALLY == False):
            devices = tf.config.experimental.list_physical_devices(device_type='CPU')
            tf.config.experimental.set_visible_devices(devices=devices, device_type='CPU')        
            TFModelBase._IS_FORCE_CPU_GLOBALLY = True

    def predict(self, features):
        if(self.is_force_cpu or self._IS_FORCE_CPU_GLOBALLY):
            with tf.device('/CPU'):
                y_predict = self._predict_with_limit_size(features, TFModelBase._MAX_CPU_FEATURE_SIZE)
        else:
            y_predict = self._predict_with_limit_size(features, TFModelBase._MAX_GPU_FEATURE_SIZE)

        return y_predict
    '''=============================================================================================================='''
    ''' One input'''
    def predict_1input(self, features):
        if(self.is_force_cpu or self._IS_FORCE_CPU_GLOBALLY):
            with tf.device('/CPU'):
                if self.TFlite:
                    y_predict = self._predict_with_limit_size_TFlite(features, TFModelBase._MAX_CPU_FEATURE_SIZE)
                else:
                    y_predict = self._predict_with_limit_size(features, TFModelBase._MAX_CPU_FEATURE_SIZE)
        else:
            if self.TFlite:
                y_predict = self._predict_with_limit_size_TFlite(features, TFModelBase._MAX_GPU_FEATURE_SIZE)
            else:
                y_predict = self._predict_with_limit_size(features, TFModelBase._MAX_GPU_FEATURE_SIZE)

        return y_predict

    def predict1_1input(self, features):
        if(self.is_force_cpu or self._IS_FORCE_CPU_GLOBALLY):
            with tf.device('/CPU'):
                if self.TFlite:
                    y_predict = self._predict_with_limit_size_TFlite(features, TFModelBase._MAX_CPU_FEATURE_SIZE)
                else:
                    y_predict = self._predict1_with_limit_size(features, TFModelBase._MAX_CPU_FEATURE_SIZE)
        else:
            if self.TFlite:
                y_predict = self._predict_with_limit_size_TFlite(features, TFModelBase._MAX_GPU_FEATURE_SIZE)
            else:
                y_predict = self._predict1_with_limit_size(features, TFModelBase._MAX_GPU_FEATURE_SIZE)

        return y_predict
    ''' Two inputs'''
    def predict_2inputs(self, features):
        if(self.is_force_cpu or self._IS_FORCE_CPU_GLOBALLY):
            with tf.device('/CPU'):
                y_predict = self._predict_with_limit_size_2inputs([features[0],features[1]], TFModelBase._MAX_CPU_FEATURE_SIZE)
        else:
            y_predict = self._predict_with_limit_size_2inputs([features[0],features[1]], TFModelBase._MAX_GPU_FEATURE_SIZE)

        return y_predict

    def _predict_with_limit_size_2inputs(self, features, max_size):

        split_size = int(np.ceil(features[0].size / max_size))
        if(split_size <= 1):
            y_predict = self.model_sub.predict(features)
        else: # too large so we have to process it separately to avoid overloading GPU
            step = int(features[0].shape[0] / split_size)
            step = max(1, step)
            y_predict = self.model_sub.predict([features[0][0:step],features[1][0:step]])
            for i in range(step, features[0].shape[0], step):
                end = min(features[0].shape[0], i + step)
                temp = self.model_sub.predict([features[0][i:end],features[1][i:end]])
                y_predict = self._append_predict_results(y_predict, temp)
        return y_predict

    '''=============================================================================================================='''

    ''' One input: TFlite'''
    def _predict_with_limit_size_TFlite(self, features, max_size):
        ''' Process Frame by Frame'''
        y_predict = self.predict_TFlite(features[0])
        for iBatch in range(1,features.shape[0]):
            temp = self.predict_TFlite(features[iBatch])
            y_predict = self._append_predict_results(y_predict, temp)
        return y_predict

    def _predict_with_limit_size(self, features, max_size):

        split_size = int(np.ceil(features.size / max_size))
        if(split_size <= 1):
            y_predict = self.model.predict(features)
        else: # too large so we have to process it separately to avoid overloading GPU
            step = int(features.shape[0] / split_size)
            step = max(1, step)
            y_predict = self.model.predict(features[0:step])
            for i in range(step, features.shape[0], step):
                end = min(features.shape[0], i + step)
                temp = self.model.predict(features[i:end])
                y_predict = self._append_predict_results(y_predict, temp)
        return y_predict

    def _predict1_with_limit_size(self, features, max_size):

        split_size = int(np.ceil(features.size / max_size))
        if(split_size <= 1):
            y_predict = self.model_sub.predict(features)
        else: # too large so we have to process it separately to avoid overloading GPU
            step = int(features.shape[0] / split_size)
            step = max(1, step)
            y_predict = self.model_sub.predict(features[0:step])
            for i in range(step, features.shape[0], step):
                end = min(features.shape[0], i + step)
                temp = self.model_sub.predict(features[i:end])
                y_predict = self._append_predict_results(y_predict, temp)
        return y_predict

    @abc.abstractmethod
    def _append_predict_results(self, first, second):                    
        raise NotImplementedError

    def predict_TFlite(self, features):
        '''
        Predict the probability of ...
        '''
        features_input = features.astype(np.float32)

        self.model.set_tensor(self.input_detail['index'], [features_input])
        self.model.invoke()

        onset_conf = self.model.get_tensor(self.Onset['index']).squeeze()
        dur_conf = self.model.get_tensor(self.Duration['index']).squeeze()
        pitch_conf = self.model.get_tensor(self.Pitch['index']).squeeze()
        return onset_conf, dur_conf, pitch_conf
