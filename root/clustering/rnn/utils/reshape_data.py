import tensorflow as tf
import numpy as np

def reshape_subjects_time_series(subjects_time_series):
    max_len_image = np.max([len(i) for i in subjects_time_series])
    #print(max_len_image)

    subjects_time_series_reshaped = []
    for subject_data in subjects_time_series:
        # Padding
        N = max_len_image - len(subject_data)
        padded_array = np.pad(subject_data, ((0, N), (0, 0)),
                              'constant', constant_values=(0))
        subject_data = padded_array
        subject_data = np.array(subject_data)
        subject_data.reshape(subject_data.shape[0], subject_data.shape[1], 1)
        subjects_time_series_reshaped.append(subject_data)

    #print(np.array(subjects_time_series_reshaped).shape)

    return subjects_time_series_reshaped

def reshape_subjects_time_series_train_test(subjects_time_series, subjects_time_series_train, subjects_time_series_test):
    #Reshapes data to 4D for Hierarchical RNN.
    t_shape = np.array(subjects_time_series).shape[1]
    RSN_shape = np.array(subjects_time_series).shape[2]

    subjects_time_series_train = np.reshape(subjects_time_series_train, (len(subjects_time_series_train), t_shape, RSN_shape))
    subjects_time_series_test = np.reshape(subjects_time_series_test, (len(subjects_time_series_test), t_shape, RSN_shape))

    subjects_time_series_train = subjects_time_series_train.astype('float32')
    subjects_time_series_test = subjects_time_series_test.astype('float32')

    return subjects_time_series_train, subjects_time_series_test

def classes_to_one_hot_encoding(classes_train, classes_test):
    # Converts class vectors to binary class matrices.
    classes_train_one_hot = tf.keras.utils.to_categorical(classes_train, 2)
    classes_test_one_hot = tf.keras.utils.to_categorical(classes_test, 2)
    return classes_train_one_hot, classes_test_one_hot

def classes_to_one_hot_encoding_negative(y_negatives):
    # Converts class vectors to binary class matrices.
    # from sklearn.preprocessing import OneHotEncoder
    # enc = OneHotEncoder(categories='auto', sparse=False)
    # print("y_negatives.reshape([-1, 1])")
    # # print(y_negatives.reshape([-1, 1]))
    # onehot_sklearn = enc.fit_transform(y_negatives.reshape([-1, 1]))

    ll = []
    for y in y_negatives:
        if(y>0):
            ll.append(1)
        else:
            ll.append(0)

    classes = np.array(ll)
    classes_test_one_hot = tf.keras.utils.to_categorical(classes, 2)

    return classes_test_one_hot

def one_hot_to_indices(data):
    print(data)
    indices = []
    for el in data:
        indices.append(list(el).index(1))
    return indices


def classes_to_normal_encoding(classes_one_hot):
    # Converts class vectors to binary class matrices.
    indices = one_hot_to_indices(classes_one_hot)

    print(indices)
    return indices


def prob2d_to_1d(prob2d):
    # Converts class vectors to binary class matrices.
    return prob2d[:,0]

