import pickle

from clustering.rnn.fit_rnn_model import fit_rnn_model
from clustering.rnn.generate_rnn_model import generate_rnn_model
from clustering.rnn.utils.data_augmentation import data_augmentation
from clustering.rnn.utils.generate_results_nn import generate_loss_accuracy_plots_nn, \
    generate_metrics_nn
from clustering.rnn.utils.generate_training_test_set import generate_training_test_set
from clustering.rnn.utils.reshape_data import reshape_subjects_time_series_train_test, reshape_subjects_time_series, \
    classes_to_one_hot_encoding
from clustering.rnn.utils.utils import generate_model_name, save_history
from utils.load_time_series_and_classes import load_time_series_and_classes


def rnn(TIME_SERIES_FOLDER, CNN, LSTM, GRU):
    subjects_time_series, classes = load_time_series_and_classes(TIME_SERIES_FOLDER)
    subjects_time_series = reshape_subjects_time_series(subjects_time_series)

    subjects_time_series_train, subjects_time_series_test, classes_train, classes_test = \
        generate_training_test_set(subjects_time_series, classes)

    subjects_time_series_train, subjects_time_series_test, classes_train, classes_test =\
        data_augmentation(subjects_time_series_train, subjects_time_series_test, classes_train, classes_test)

    subjects_time_series_train, subjects_time_series_test = \
        reshape_subjects_time_series_train_test(subjects_time_series, subjects_time_series_train, subjects_time_series_test)

    classes_train_one_hot, classes_test_one_hot = classes_to_one_hot_encoding(classes_train, classes_test)

    model = generate_rnn_model(subjects_time_series, CNN, LSTM, GRU)

    model, history = fit_rnn_model(model, subjects_time_series_train, subjects_time_series_test, classes_train_one_hot)

    model_name = generate_model_name(CNN, LSTM, GRU)
    model_filepath = model_name
    model.save(model_name)
    history_filename = save_history(model_name, history)

    generate_loss_accuracy_plots_nn(history_filename)
    generate_metrics_nn(model_filepath, subjects_time_series_test, classes_test_one_hot)
