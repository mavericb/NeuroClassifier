import os
import numpy


def load_time_series_and_classes(TIME_SERIES_FOLDER):
    # LOAD TIME-SERIES
    time_series_filenames = os.listdir(TIME_SERIES_FOLDER)
    try:
        time_series_filenames.remove(".DS_Store")
    except Exception as e:
        print(e)

    time_series_abs_paths = []
    for time_series in time_series_filenames:
        time_series_abs_paths.append(TIME_SERIES_FOLDER + time_series)

    pooled_subjects = []
    for time_series_abs_path in time_series_abs_paths:
        pooled_subjects.append(numpy.load(time_series_abs_path))

    classes_list = []
    for time_series_filename in time_series_filenames:
        #print(time_series_filename)
        if "sub-1" in time_series_filename:
            # classes_list.append(1)
            classes_list.append(0)  # control
        if "sub-5" in time_series_filename:
            # classes_list.append(5)
            classes_list.append(1)  # schizo

    classes = numpy.asarray(classes_list)

    return pooled_subjects, classes

