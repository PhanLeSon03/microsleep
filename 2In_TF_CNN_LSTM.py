# %%
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
from tensorflow.keras.layers import TimeDistributed, Concatenate, MaxPooling1D, LSTM
from libs.tf_data_record import TFMultiOutParserMIMO
from libs.tf_utils import TFUtils
from libs.MSE_onset import MSEOnset
from libs.misc import Misc
import sklearn
from tensorflow.keras.models import load_model
from tensorflow.keras import initializers
from tensorflow.python.ops import array_ops


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')

batch_size = 8
step_frame = MSEOnset.STEP_FRAME
feature_size = MSEOnset.N_FEATURE_FFT
channels = MSEOnset.CHANNLES
shuffle_buff = batch_size

path = './content/'
pretrain_dir = path + "MSE_Onset_Model/"

filename_best = 'best_MSE_Onset.h5' 
vocal_onset_model_path = pretrain_dir + filename_best


checkpoint_dir = path + "MSE_Onset_Model_2In/"
save_filter = "MSE_Onset"

# make sure we get the same random weight init on any run
tf.random.set_seed(1)


train_data_path = path + 'MSE_Onset_Database_2In/train/'
val_data_path = path + 'MSE_Onset_Database_2In/validation/'


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
x2_shape = [MSEOnset.PICKING_FRAME_SIZE, 2*MSEOnset.N_EEG + 2*MSEOnset.N_EOG]
x_shape = {'x1': x1_shape, 'x2': x2_shape}

y1_shape = [MSEOnset.PICKING_FRAME_SIZE, MSEOnset.N_CLASS]
y_shape = {'y1': y1_shape}

train_dataset = tf.data.TFRecordDataset(train_file_paths, compression_type="GZIP")
train_dataset = train_dataset.map(TFMultiOutParserMIMO.get_2in_out_parser(x_shape, y_shape), num_parallel_calls=5)
train_dataset = train_dataset.shuffle(shuffle_buff, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.TFRecordDataset(valid_file_paths, compression_type="GZIP")
val_dataset = val_dataset.map(TFMultiOutParserMIMO.get_2in_out_parser(x_shape, y_shape))
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)


inputs = keras.Input(shape=(MSEOnset.PICKING_FRAME_SIZE*MSEOnset.STEP_FRAME, feature_size, channels), name="input")
inputs1 = keras.Input(shape=(MSEOnset.PICKING_FRAME_SIZE, 2*MSEOnset.N_EEG + 2*MSEOnset.N_EOG), name="input1")

#----------------------------------- Conv block 1
cnn1 = Conv2D(16, (3, 3),
                              kernel_initializer='he_uniform',
                              padding='same',
                              strides=(1, 1),
                              data_format='channels_last', name='stft_conv1')(inputs)
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
# cnn_onset = (MaxPooling2D(pool_size=(1, 2)))(cnn_onset)
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
                              
                              
# ----------------- Reshape CNN to feed into RNN ------------------------------
[nS, nT, nF, nC] = cnn_onset.get_shape()
cnn_onset = tf.reshape(cnn_onset, [-1,  nT, nC*nF])
                              
                                                      
rnn_com = (GRU(32,
                            kernel_regularizer=keras.regularizers.l1_l2(0.001),
                            return_sequences=True, name='rnn1'))(cnn_onset)
rnn_com = Dropout(0.2)(rnn_com)

rnn_com = (GRU(32,
                            kernel_regularizer=keras.regularizers.l1_l2(0.001),
                            return_sequences=True, name='rnn2'))(rnn_com)
rnn_com = Dropout(0.2)(rnn_com)

# ----------------- Second input ------------------------------   
rnn_vector = Dense(32, activation='linear', name='MSE_proj')(inputs1)
rnn_vector = GRU(units=16, return_sequences=True, name='gru_vector1')(rnn_vector)
rnn_vector = Dropout(0.2, name='feat_drop1')(rnn_vector)
rnn_vector = GRU(units=8, return_sequences=True, name='gru_vector2')(rnn_vector)
rnn_vector = Dropout(0.2, name='feat_drop2')(rnn_vector)

                              
concat = Concatenate(name='concat')([rnn_com, rnn_vector])


dense_com = (Dense(32,
                                 activation='linear',
                                 kernel_regularizer=keras.regularizers.l2(0.001),
                                 bias_regularizer=keras.regularizers.l2(),
                                 name='MSE_onset_mf_0'))(concat)
                              

rnn_onset = (Dense(MSEOnset.N_CLASS,
                               activation=tf.nn.sigmoid,
                               kernel_regularizer=keras.regularizers.l2(0.001),
                               bias_regularizer=keras.regularizers.l2(),
                               name='MSE_onset_mf_1'))(dense_com)


onset_mf = Flatten(name='onset_mf')(rnn_onset)

#----------------------------------------------------------------------------

# Output layer
model = keras.Model(inputs=[inputs, inputs1],  outputs=onset_mf)

model.summary()

# %% Load init params
# model.load_weights(vocal_onset_model_path, by_name=True, skip_mismatch=True)


cl_w = [0.28157203, 3.57512078, 35.6736615, 7.10387778]
w0 = 0.2
w1 = 5

def weighted_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    # compute standard BCE
    bce = keras.backend.binary_crossentropy(y_true, y_pred)

    # apply weights
    weights = y_true * w1 + (1 - y_true) * w0
    weighted_bce = weights * bce

    return tf.reduce_mean(weighted_bce)

@tf.function
def onset_weighted_binary_crossentropy(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    error = tf.keras.backend.switch(tf.equal(y_true,0),0.1*error,error)
    onset_loss = tf.reduce_mean(error)
    l2_loss = 0.0001* sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables(scope=None))
    return onset_loss + l2_loss


            
learning_rate = 0.001
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  
    loss={
          "onset_mf": keras.losses.BinaryCrossentropy(label_smoothing=0.0),#weighted_binary_crossentropy, #
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


model.save(checkpoint_dir + filename_best, save_format="h5")


latest_file = TFUtils.get_latest_filepath(checkpoint_dir, save_filter)
model.load_weights(latest_file, by_name=True, skip_mismatch=False)
print(latest_file)

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

