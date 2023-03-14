from sklearn.model_selection import train_test_split
import constants


def generate_training_test_set(subjects_time_series, classes):
    subjects_time_series_train, subjects_time_series_test, classes_train, classes_test = \
        train_test_split(subjects_time_series, classes, test_size=constants.SPLIT_SIZE, random_state=constants.RANDOM_STATE)

    return subjects_time_series_train, subjects_time_series_test, classes_train, classes_test