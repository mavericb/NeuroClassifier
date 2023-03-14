import time
import numpy
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import constants
from clustering.svm.svm import svm
from utils.load_time_series_and_classes import load_time_series_and_classes


def multiple_svm(TIME_SERIES_FOLDER):
    # CLUSTERING VIA SVM + (stratified)CROSS-VALIDATION
    kinds = ['correlation', 'partial correlation', 'tangent']
    n_splits = constants.N_SPLITS
    test_size = constants.TEST_SIZE

    subjects_time_series, classes = load_time_series_and_classes(TIME_SERIES_FOLDER)
    cv = StratifiedShuffleSplit(n_splits=n_splits, random_state=0, test_size=test_size)
    subjects_time_series = numpy.asarray(subjects_time_series)

    print("## DATA ##")
    print("dataset_size: ", len(classes))
    print("test_size :", test_size)
    print("n_splits: ", n_splits, "\n")

    accuracies = {}
    sensitivities = {}
    specificities = {}
    precisions = {}
    f1s = {}
    kappas = {}
    roc_aucs = {}

    for kind in kinds:
        t = time.time()
        print("## ", kind, " ##")

        accuracies[kind] = []
        sensitivities[kind] = []
        specificities[kind] = []
        precisions[kind] = []
        f1s[kind] = []
        kappas[kind] = []
        roc_aucs[kind] = []

        for train, test in cv.split(subjects_time_series, classes):
            print("pooled_subjects")
            # print(X)
            print((subjects_time_series.shape))
            print("classes")
            print(classes)
            print((classes.shape))
            print("train")
            # print(y_prob)
            print((train.shape))
            print("test")
            # print(y_prob)
            print((test.shape))


            accuracy, sensitivity, specificity, precision, f1, kappa, roc_auc = \
                svm(kind, subjects_time_series, classes, train, test)
            accuracies[kind].append(accuracy)
            sensitivities[kind].append(sensitivity)
            specificities[kind].append(specificity)
            precisions[kind].append(precision)
            f1s[kind].append(f1)
            kappas[kind].append(kappa)
            roc_aucs[kind].append(roc_auc)

        elapsed = time.time() - t
        print("clustering - stopsignal - svm-",kind,": ", elapsed)


    mean_accuracy = [np.mean(accuracies[kind]) for kind in kinds]
    accuracy_std = [np.std(accuracies[kind]) for kind in kinds]
    mean_sensitivity = [np.mean(sensitivities[kind]) for kind in kinds]
    sensitivity_std = [np.std(sensitivities[kind]) for kind in kinds]
    mean_specificity = [np.mean(specificities[kind]) for kind in kinds]
    specificity_std = [np.std(specificities[kind]) for kind in kinds]

    mean_precision = [np.mean(precisions[kind]) for kind in kinds]
    precision_std = [np.std(precisions[kind]) for kind in kinds]
    mean_f1 = [np.mean(f1s[kind]) for kind in kinds]
    f1_std = [np.std(f1s[kind]) for kind in kinds]
    mean_kappa = [np.mean(kappas[kind]) for kind in kinds]
    kappa_std = [np.std(kappas[kind]) for kind in kinds]

    mean_roc_auc = [np.mean(roc_aucs[kind]) for kind in kinds]
    roc_auc_std = [np.std(roc_aucs[kind]) for kind in kinds]

    print("mean\t \t ", kinds,'\t -- \t std ', kinds)
    print("accuracy", mean_accuracy, accuracy_std)
    print("sensitivity", mean_sensitivity, sensitivity_std)
    print("specificity", mean_specificity, specificity_std)
    print("precision", mean_precision, precision_std)
    print("f1", mean_f1, f1_std)
    print("kappa", mean_kappa, kappa_std)
    print("roc_auc", mean_roc_auc, roc_auc_std)
