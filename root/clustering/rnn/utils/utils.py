import datetime
import pickle

def generate_model_name(CNN, LSTM, GRU):
    model_name = 'model'
    if CNN: model_name += '-CNN'
    if LSTM: model_name += '-LSTM'
    if GRU: model_name += '-GRU'
    model_name += '-' + str(datetime.datetime.now()) + '.h5'
    return model_name

def save_history(model_name, history):
    history_filename = model_name+"-history.pkl"
    with open(history_filename, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    return history_filename

def generate_model_transformer():
    model_name = 'model-Transformer'
    model_name += '-' + str(datetime.datetime.now()) + '.h5'
    return model_name