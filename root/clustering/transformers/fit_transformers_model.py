from keras.callbacks import EarlyStopping, ReduceLROnPlateau

def fit_transformer_model(model, X_train, y_train, X_test):
    verbose = False

    if verbose:
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

    callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True),
                 ReduceLROnPlateau(monitor='val_loss', mode='min', patience=5, factor=0.1, min_lr=0.00001)]

    #print(y_train)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,#0.2 ensemble
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )

    return model, history