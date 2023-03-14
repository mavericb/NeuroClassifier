from clustering.rnn.utils.data_augmentation import data_augmentation
from clustering.rnn.utils.generate_results_nn import generate_loss_accuracy_plots_nn, generate_metrics_nn
from clustering.rnn.utils.generate_training_test_set import generate_training_test_set
from clustering.rnn.utils.reshape_data import reshape_subjects_time_series, reshape_subjects_time_series_train_test, \
    classes_to_one_hot_encoding
from clustering.rnn.utils.utils import generate_model_transformer, save_history
from clustering.transformers.fit_transformers_model import fit_transformer_model
from clustering.transformers.generate_transformers_model import generate_tranformer_model
from utils.load_time_series_and_classes import load_time_series_and_classes


def transformers(TIME_SERIES_FOLDER):
    subjects_time_series, classes = load_time_series_and_classes(TIME_SERIES_FOLDER)
    subjects_time_series = reshape_subjects_time_series(subjects_time_series)

    subjects_time_series_train, subjects_time_series_test, classes_train, classes_test = \
        generate_training_test_set(subjects_time_series, classes)

    subjects_time_series_train, subjects_time_series_test, classes_train, classes_test = \
        data_augmentation(subjects_time_series_train, subjects_time_series_test, classes_train, classes_test)

    subjects_time_series_train, subjects_time_series_test = \
        reshape_subjects_time_series_train_test(subjects_time_series, subjects_time_series_train, subjects_time_series_test)

    classes_train_one_hot, classes_test_one_hot = classes_to_one_hot_encoding(classes_train, classes_test)

    model = generate_tranformer_model(subjects_time_series_train)
    model, history = fit_transformer_model(model, subjects_time_series_train, classes_train_one_hot, subjects_time_series_test)

    model_name = generate_model_transformer()
    model_filepath = model_name
    model.save(model_name)
    history_filename = save_history(model_name, history)

    generate_loss_accuracy_plots_nn(history_filename)
    generate_metrics_nn(model_filepath, subjects_time_series_test, classes_test_one_hot)