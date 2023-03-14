from nilearn.connectome import ConnectivityMeasure
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    cohen_kappa_score, roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def svm(kind, pooled_subjects,classes, train, test):
    # *ConnectivityMeasure* can output the estimated subjects coefficients
    # as a 1D arrays through the parameter *vectorize*.
    connectivity = ConnectivityMeasure(kind=kind, vectorize=True)

    # build vectorized connectomes for subjects in the train set
    print("pooled_subjects[train]")
    print(pooled_subjects[train])
    print(pooled_subjects[train].shape)
    connectomes = connectivity.fit_transform(pooled_subjects[train])

    # fit the classifier
    classifier = CalibratedClassifierCV(LinearSVC())
    CalibratedClassifierCV(base_estimator=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                                                    intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                                                    multi_class='ovr', penalty='l2', random_state=None,
                                                    tol=0.0001,
                                                    verbose=0),
                           cv=3, method='sigmoid')
    classifier.fit(connectomes, classes[train])

    # make predictions for the left-out test subjects
    predictions_probs = classifier.predict_proba(
        connectivity.transform(pooled_subjects[test]))[:, 1]
    threshold = 0.5
    predictions = np.where(predictions_probs > threshold, 1, 0)

    # confusion matrix
    CM = confusion_matrix(classes[test], predictions)
    print(CM)

    print("classes[test]\n",classes[test])
    print("predictions\n", predictions)
    print("CM", CM)

    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    print(TN, FN, TP, FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    accuracy = ACC
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    sensitivity = TPR
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    specificity = TNR
    # Precision or positive predictive value (replaced with precision_score because it can lead to nan)
    # PPV = TP / (TP + FP)
    # precision = PPV

    # redundant
    # # accuracy: (tp + tn) / (p + n)
    # accuracy = accuracy_score(classes[test], predictions)
    # print('Accuracy: %f' % accuracy)
    #
    # # recall or sensitivity: tp / (tp + fn)
    # sensitivity = recall_score(classes[test], predictions)
    # print('Recall: %f' % sensitivity)
    #
    # # precision tp / (tp + fp)
    precision = precision_score(classes[test], predictions)
    # print('Precision: %f' % precision)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(classes[test], predictions)
    print('F1 score: %f' % f1)

    # kappa
    kappa = cohen_kappa_score(classes[test], predictions)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(classes[test], predictions_probs)
    print('ROC AUC: %f' % auc)

    print("predictions_probs")
    print(predictions_probs)

    return accuracy, sensitivity, specificity, precision, f1, kappa, auc
