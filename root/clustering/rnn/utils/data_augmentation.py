from imblearn.over_sampling import SMOTE
import numpy as np

import constants


def data_augmentation(X_train, X_test, classes_train, classes_test):
    ## data augmentation - training set
    sm = SMOTE(random_state=constants.SAMPLING_RANDOM_STATE)

    nsamples, nx, ny = np.asarray(X_train).shape
    X_train = np.asarray(X_train).reshape((nsamples, nx * ny))
    X_train, classes_train = sm.fit_resample(X_train, classes_train)


    ## data augmentation - test set
    sm = SMOTE(constants.SAMPLING_RANDOM_STATE)

    nsamples, nx, ny = np.asarray(X_test).shape
    X_test = np.asarray(X_test).reshape((nsamples, nx * ny))
    X_test, classes_test = sm.fit_resample(X_test, classes_test)

    return X_train, X_test, classes_train, classes_test
