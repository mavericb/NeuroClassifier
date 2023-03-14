import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution1D, LSTM, GRU


def add_cnn(tf, model, t_shape, RSN_shape):
    # 3d convolution
    # https://github.com/bsplku/3dcnn4fmri/blob/master/Python_code/3dcnn_fmri_demo.ipynb
    model.add(Convolution1D(input_shape=(t_shape, RSN_shape),
                            filters=32,
                            kernel_size=(3),
                            activation=tf.nn.relu))

def add_LSTM(model, t_shape, RSN_shape):
    model.add(LSTM(units=70,  # dimensionality of the output space
                   dropout=0.4,  # Fraction of the units to drop (inputs)
                   recurrent_dropout=0.15,  # Fraction of the units to drop (recurent state)
                   return_sequences=True,  # return the last state in addition to the output
                   input_shape=(t_shape, RSN_shape)))

    model.add(LSTM(units=60,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=True))

    model.add(LSTM(units=50,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=True))

    model.add(LSTM(units=40,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=False))

def add_GRU(model, t_shape, RSN_shape):
    model.add(GRU(units=70,  # dimensionality of the output space
                  dropout=0.4,  # Fraction of the units to drop (inputs)
                  recurrent_dropout=0.15,  # Fraction of the units to drop (recurent state)
                  return_sequences=True,  # return the last state in addition to the output
                  input_shape=(t_shape, RSN_shape)))

    model.add(GRU(units=60,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=True))

    model.add(GRU(units=50,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=True))

    model.add(GRU(units=40,
                   dropout=0.4,
                   recurrent_dropout=0.15,
                   return_sequences=False))


def generate_rnn_model(subjects_time_series, CNN, LSTM, GRU):
    model = Sequential()
    t_shape = np.array(subjects_time_series).shape[1]
    RSN_shape = np.array(subjects_time_series).shape[2]

    if (CNN):
        add_cnn(tf, model, t_shape, RSN_shape)
    if (LSTM):
        add_LSTM(model, t_shape, RSN_shape)
    if (GRU):
        add_GRU(model, t_shape, RSN_shape)

    model.add(Dense(units=2,
                    activation="sigmoid"))

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['binary_accuracy'])

    print(model.summary())
    return model

def custom_loss_function_square(y_true, y_pred):
   squared_difference = tf.square(y_true - y_pred)
   return tf.reduce_mean(squared_difference, axis=-1)
    # + model.compile(loss=customLoss, optimizer='sgd')


def generate_rnn_model_custom(subjects_time_series, CNN, LSTM, GRU,  subjects_time_series_i, classes_i):
    model = Sequential()
    print(subjects_time_series)
    print(subjects_time_series.shape)
    t_shape = np.array(subjects_time_series).shape[1]
    RSN_shape = np.array(subjects_time_series).shape[2]

    if (CNN):
        add_cnn(tf, model, t_shape, RSN_shape)
    if (LSTM):
        add_LSTM(model, t_shape, RSN_shape)
    if (GRU):
        add_GRU(model, t_shape, RSN_shape)

    model.add(Dense(units=2,
                    activation="sigmoid"))


    model.compile(loss=tf.keras.losses.Hinge(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['binary_accuracy'])

    print(model.summary())
    return model


