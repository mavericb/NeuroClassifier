from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def fit_rnn_model(model, subjects_time_series_train, subjects_time_series_test, classes_train_one_hot):
    verbose = False

    if verbose:
        print(subjects_time_series_train.shape[0], 'train samples')
        print(subjects_time_series_test.shape[0], 'test samples')

    history = model.fit(subjects_time_series_train, classes_train_one_hot, validation_split=0.1, epochs=100,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.5, min_lr=1e-5)
                        ])

    return model, history

def fit_rnn_model_s(model, subjects_time_series_train, classes_train_one_hot):

    history = model.fit(subjects_time_series_train, classes_train_one_hot, validation_split=0.2, epochs=100,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.1, min_lr=1e-5)
                        ])

    return model, history

def fit_rnn_model_short(model, subjects_time_series_train, classes_train_one_hot):
    # verbose = False
    #
    # if verbose:
    #     print(subjects_time_series_train.shape[0], 'train samples')
    #     print(subjects_time_series_test.shape[0], 'test samples')

    history = model.fit(subjects_time_series_train, classes_train_one_hot, validation_split=0.1, epochs=30,
                        callbacks=[
                            EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', mode='min', patience=10, factor=0.3 , min_lr=1e-5)
                        ])
    return model, history


