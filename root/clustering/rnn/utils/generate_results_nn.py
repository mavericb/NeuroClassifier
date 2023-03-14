import datetime

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, cohen_kappa_score, f1_score, \
    precision_score, recall_score


def generate_loss_accuracy_plots_nn(history_filename):
    #LOSS and ACCURACY
    # summarize history for accuracy
    import pickle
    f = history_filename
    history = pickle.load(open(f, "rb"))

    plt.plot(history['binary_accuracy'])
    plt.plot(history['val_binary_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    #plt.show()
    plt.savefig(str(datetime.datetime.now()) + ".png")
    plt.close()

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    #plt.show()
    plt.savefig(str(datetime.datetime.now())+".png")
    plt.close()

def generate_metrics_nn(model_filepath, subjects_time_series_test, classes_test_one_hot):
    with open(str(datetime.datetime.now()) + '.txt', 'w') as f: #TODO handle output in and/or file
        # METRICS
        # model.fit(X_train, y_train, epochs=30)
        model = keras.models.load_model(model_filepath)

        print('evaluating..')
        y_pred = model.predict(subjects_time_series_test)
        # print(y_pred)
        # print(classes_test)

        # y_test_1d = [i[0] for i in classes_test_one_hot]
        # y_pred_1d = [1.0 if i[0] > .5 else 0.0 for i in y_pred]
        # y_test_1d = [i[1] for i in classes_test_one_hot]
        # y_pred_1d = [1.0 if i[1] > .5 else 0.0 for i in y_pred]

        print("y_pred")
        print(y_pred)
        y_test_1d = [i[1] for i in classes_test_one_hot]
        y_pred_1d = [1.0 if i[1] > .5 else 0.0 for i in y_pred]

        print("y_test_1d")
        print(y_test_1d)

        print("y_pred_1d")
        print(y_pred_1d)

        # confusion matrix
        CM = confusion_matrix(y_test_1d, y_pred_1d)
        print("CM")
        print(CM)

        # print("classes[test]\n",classes[test])
        # print("predictions\n", predictions)
        # print("CM", CM)

        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]

        print("[[TN,FP],[FN, TP]]")

        print("TN, FN, TP, FP")
        print(TN, FN, TP, FP)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        accuracy = ACC
        print("accuracy")
        print(accuracy)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        sensitivity = TPR
        print("sensitivity")
        print(sensitivity)

        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        specificity = TNR
        print("specificity")
        print(specificity)
        # Precision or positive predictive value (replaced with precision_score because it can lead to nan)
        # PPV = TP / (TP + FP)
        # precision = PPV

        # redundant
        # accuracy: (tp + tn) / (p + n)
        accuracy = accuracy_score(y_test_1d, y_pred_1d)
        print('Accuracy: %f' % accuracy)
        #
        # recall or sensitivity: tp / (tp + fn)
        sensitivity = recall_score(y_test_1d, y_pred_1d)
        print('sensitivity: %f' % sensitivity)
        #
        # # precision tp / (tp + fp)

        precision = precision_score(y_test_1d, y_pred_1d)
        print('Precision: %f' % precision)

        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(y_test_1d, y_pred_1d)
        print('F1 score: %f' % f1)

        # kappa
        kappa = cohen_kappa_score(y_test_1d, y_pred_1d)
        print('Cohens kappa: %f' % kappa)
        # ROC AUC
        auc = roc_auc_score(y_test_1d, y_pred_1d)
        print('ROC AUC: %f' % auc)


        return accuracy, sensitivity, specificity, precision, f1, kappa, auc