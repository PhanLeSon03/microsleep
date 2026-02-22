import pathlib
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as kb
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, ELU, Input, SeparableConv2D, MaxPooling2D, add, Flatten
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Bidirectional, GRU, Dropout, multiply
from tensorflow.keras.layers import TimeDistributed, Concatenate, MaxPooling1D, LSTM, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Add, Lambda
from libs.tf_data_record import TFMultiOutParserMIMO
from libs.tf_utils import TFUtils
from libs.MSE_onset import MSEOnset
from libs.misc import Misc
import sklearn
from tensorflow.keras.models import load_model
from tensorflow.keras import initializers
from tensorflow.python.ops import array_ops
from sklearn.metrics import cohen_kappa_score


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')

batch_size = 8
step_frame = MSEOnset.STEP_FRAME
feature_size = MSEOnset.N_FEATURE_FFT
channels = MSEOnset.CHANNLES
shuffle_buff = batch_size

path = './content/'
pretrain_dir = path + "MSE_Onset_Model_W/"

filename_best = 'best_MSE_Onset.h5' 
vocal_onset_model_path = pretrain_dir + filename_best


checkpoint_dir = path + "MSE_Onset_Model_W/"
save_filter = "MSE_Onset"

# make sure we get the same random weight init on any run
tf.random.set_seed(1)


train_data_path = path + 'MSE_Onset_Database_W/train/'
val_data_path = path + 'MSE_Onset_Database_W/validation/'


train_names = os.listdir(train_data_path)
train_file_paths = [
    (train_data_path + file_name)
    for file_name in train_names
]

np.random.shuffle(train_file_paths)
valid_names = os.listdir(val_data_path)
valid_file_paths = [
    (val_data_path + file_name)
    for file_name in valid_names
]


x1_shape = [MSEOnset.PICKING_FRAME_SIZE*step_frame, MSEOnset.N_FEATURE_FFT, channels]
x2_shape = [MSEOnset.PICKING_FRAME_SIZE, MSEOnset.WIN_WAVE, channels]
x_shape = {'x1': x1_shape, 'x2': x2_shape}

y1_shape = [MSEOnset.PICKING_FRAME_SIZE, MSEOnset.N_CLASS]
y_shape = {'y1': y1_shape}

train_dataset = tf.data.TFRecordDataset(train_file_paths, compression_type="GZIP")
train_dataset = train_dataset.map(TFMultiOutParserMIMO.get_2in_out_parser(x_shape, y_shape))
train_dataset = train_dataset.shuffle(shuffle_buff, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.TFRecordDataset(valid_file_paths, compression_type="GZIP")
val_dataset = val_dataset.map(TFMultiOutParserMIMO.get_2in_out_parser(x_shape, y_shape))
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


class GroupOrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, channels: int, nFilter: int, lmbda: float = 1e-3):
        self.channels = channels
        self.nFilter = nFilter
        self.lmbda = lmbda

    def __call__(self, w):
        # Conv1D: (size, in, out) = (301, 4, 16*4)
        # With groups=nC: in == nC, out == nC*nF
        nK = tf.shape(w)[0]
        w = tf.reshape(w, [nK, self.channels, self.nFilter])

        # for each channel, compute Gram matrix: (nFilter, nFilter)
        # W: (nK, nC, nF) -> (nC, nK, nF)
        w_ch = tf.transpose(w, [1, 0, 2])  # (nC, nK, nF)

        # normalize each filter to unit norm (avoid scale tricks)
        w_ch = tf.math.l2_normalize(w_ch, axis=1)  # normalize over nK

        gram = tf.matmul(w_ch, w_ch, transpose_a=True)  # (ch, f, f)
        I = tf.eye(self.nFilter, batch_shape=[self.channels])
        loss = tf.reduce_mean(tf.square(gram - I))
        return self.lmbda * loss

    def get_config(self):
        return {"channels": self.channels, "nFilter": self.nFilter, "lmbda": self.lmbda}
    

spec = keras.Input(shape=(MSEOnset.PICKING_FRAME_SIZE*MSEOnset.STEP_FRAME, feature_size, channels), name="input_spec")
wave = keras.Input(shape=(MSEOnset.PICKING_FRAME_SIZE, MSEOnset.WIN_WAVE,channels), name="input_wave")

mf = TimeDistributed(
    Conv1D(
        filters= 16 * channels,
        kernel_size=155,
        use_bias=False,
        groups=channels,   
        kernel_regularizer=GroupOrthogonalRegularizer(channels, 16, lmbda=1e-3),
        name="match_filter"
    ),
    name="td_match_filter"
)(wave)   # (B, S, T, 64*C)
mf = BatchNormalization()(mf)
mf = Lambda(lambda x: tf.abs(x), name="abs_mf")(mf)
mf = TimeDistributed(GlobalMaxPooling1D(), name="td_globalmax")(mf)


#----------------------------------- Conv block 1
cnn1 = Conv2D(16, (3, 3),
                              kernel_initializer='he_uniform',
                              padding='same',
                              strides=(1, 1),
                              data_format='channels_last', name='stft_conv1')(spec)
cnn1 = (ELU())(cnn1)
cnn1 = (MaxPooling2D(pool_size=(2, 2)))(cnn1)
cnn1 = BatchNormalization(name='batch_conv1')(cnn1)
cnn1 = Dropout(0.3)(cnn1)
#----------------------------------- Conv block 2
cnn1 = Conv2D(32, (3, 3),
                              kernel_initializer='he_uniform',
                              padding='same',
                              data_format='channels_last', name='stft_conv2')(cnn1)
cnn1 = (ELU())(cnn1)
cnn1 = (MaxPooling2D(pool_size=(2, 2)))(cnn1)
cnn1 = BatchNormalization(name='batch_conv2')(cnn1)
cnn1 = Dropout(0.3)(cnn1)

#----------------------------------- Conv block 3
cnn_onset = Conv2D(64, (3, 3),
                             kernel_initializer='he_uniform',
                             padding='same',
                             data_format='channels_last', name='stft_conv3')(cnn1)
cnn_onset = (ELU())(cnn_onset)
cnn_onset = (MaxPooling2D(pool_size=(1, 2)))(cnn_onset)
cnn_onset = BatchNormalization(name='batch_conv3')(cnn_onset)
cnn_onset = Dropout(0.3)(cnn_onset)

#----------------------------------- Conv block 4
cnn_onset = Conv2D(128, (3, 3),
                             kernel_initializer='he_uniform',
                             padding='same',
                             data_format='channels_last', name='stft_conv4')(cnn_onset)
cnn_onset = (ELU())(cnn_onset)
cnn_onset = (MaxPooling2D(pool_size=(1, 2)))(cnn_onset)
cnn_onset = BatchNormalization(name='batch_conv4')(cnn_onset)
cnn_onset = Dropout(0.3)(cnn_onset)


# ----------------- RNN 
# ----------------- Reshape 
[nS, nT, nF, nC] = cnn_onset.get_shape()
rnn_com = tf.reshape(cnn_onset, [-1,  nT, nC*nF])

concat = Concatenate(name='concat')([rnn_com, mf])

rnn_com = (GRU(32,
                            kernel_regularizer=keras.regularizers.l1_l2(0.001),
                            return_sequences=True, name='rnn1'))(concat)
rnn_com = Dropout(0.2)(concat)

rnn_com = (GRU(32,
                            kernel_regularizer=keras.regularizers.l1_l2(0.001),
                            return_sequences=True, name='rnn2'))(rnn_com)
rnn_com = Dropout(0.2)(rnn_com)

                              
rnn_com = (Dense(32,
                                 activation='linear',
                                 kernel_regularizer=keras.regularizers.l2(0.001),
                                 bias_regularizer=keras.regularizers.l2(),
                                 name='MSE_onset_mf_0'))(rnn_com)

rnn_onset = (Dense(1, activation=tf.nn.sigmoid,
                      bias_regularizer=keras.regularizers.l2(),
                      name='MSE_onset_mf_1'))(rnn_com)


onset_mf = Flatten(name='onset_mf')(rnn_onset)

#----------------------------------------------------------------------------

model = keras.Model(inputs=[spec, wave],  outputs=onset_mf)

model.summary()

# %% Load init params
# model.load_weights(vocal_onset_model_path, by_name=True, skip_mismatch=True)

#===============================================
# filter_list = ['vocal_g1_conv1', 'vocal_g1_conv2','vocal_g1_conv3','vocal_onset_conv','vocal_rnn_onset_0','vocal_rnn_onset_1','vocal_onset_mf_0'
#               ,'vocal_onset_mf_1','vocal_dur_mf_1']
# for layer in model.layers:
#     # print(layer.name)
#     if(layer.name in filter_list):
#         layer.trainable = False
#     else:
#         print(layer.name)
#         layer.trainable = True
# ===============================================
# model.summary()


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent)

@tf.function
def sparse_loss(yTrue,yPred):
    loss = focal_loss(yPred, yTrue)
    return loss


w0 = 0.5
w1 = 3
def weighted_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    bce = keras.backend.binary_crossentropy(y_true, y_pred)

    weights = y_true * w1 + (1 - y_true) * w0
    weighted_bce = weights * bce

    return tf.reduce_mean(weighted_bce)

@tf.function
def onset_weighted_binary_crossentropy(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    error = tf.keras.backend.switch(tf.equal(y_true,0),w0*error,w1*error)
    onset_loss = tf.reduce_mean(error)
    l2_loss = 0.0001* sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables(scope=None))
    return onset_loss + l2_loss


            
learning_rate = 0.001
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  
    loss={
          "onset_mf":weighted_binary_crossentropy, #weighted_binary_crossentropy, #keras.losses.BinaryCrossentropy(label_smoothing=0.0), #
          },
    loss_weights={
                  "onset_mf": 1.0,
                 },
    metrics={"onset_mf": [keras.metrics.Precision(), keras.metrics.Recall()],
             },        
)

callbacks = TFUtils.get_default_callbacks(checkpoint_dir + save_filter,
                                          learn_rate_factor=0.2,
                                          reduce_learn_rate_patience=5,
                                          monitor='val_loss',
                                          stop_patience=10,
                                          log_csv_path=checkpoint_dir + 'loss.csv')

savedFiles = Misc.get_all_files(checkpoint_dir, ['.h5'])
for sf in savedFiles:
    if('best' not in sf.full_name):
        os.remove(sf.full_path.as_posix())

print("Fit model on training data")
history = model.fit(
    x=train_dataset,
    epochs=100,
    verbose=2,
    validation_data=val_dataset,
    validation_freq=1,
    callbacks=callbacks
)



latest_file = TFUtils.get_latest_filepath(checkpoint_dir, save_filter)
model.load_weights(latest_file, by_name=True, skip_mismatch=False)
print(latest_file)
model.save(checkpoint_dir + filename_best, save_format="h5")

y_true_onset = []
y_pre_onset = []
for features, label in val_dataset:
    label_pre = model.predict(features, verbose=0)
    label_exp = label[0].numpy()
    for exp, pre in zip(label_exp, label_pre[0]):        
        p = np.zeros_like(pre)
        p[pre >= 0.5] = 1   
        e = np.zeros_like(exp)
        e[exp >= 0.5] = 1
        y_true_onset.append(e)
        y_pre_onset.append(p)          

y_true_onset = np.asarray(y_true_onset).flatten()
y_pre_onset = np.asarray(y_pre_onset).flatten()
print(sklearn.metrics.classification_report(y_true_onset, y_pre_onset)) 
print('computes Cohen kappa per class')
print(TFUtils.kappa_metric(y_true_onset, y_pre_onset, n_cl = 2)) 
print(f'Cohen kappa: {cohen_kappa_score(y_true_onset, y_pre_onset)}')

