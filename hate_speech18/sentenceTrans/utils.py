# -*- coding: utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import roc_auc_score as AUC
from math import sqrt


def PRF_TPR_TNR_gmean_AUC(y_true, y_pred, y_pred_proba):
        tn, fp, fn, tp = cm(y_true, y_pred).ravel()
        if tn + fp > fn + tp:
            tpr = float(tp)/(fn+tp)
            tnr = float(tn)/(tn+fp) 
        else:
            tnr = float(tp)/(fn+tp)
            tpr = float(tn)/(tn+fp) 

        ## precision, recall, fscore
        precision = float(tp) / (tp+fp)
        recall = tpr
        gmean = sqrt(tpr*tnr)
        f1 = 2*precision*recall / (precision + recall)
        f2 = 5*precision*recall / (4*precision + recall)
        auc = AUC(y_true, y_pred_proba[:, 1])

        scores = {'gmean': gmean, 'tpr': tpr,
			'tnr': tnr, 'precision': precision, 'recall': recall,
			'f1': f1, 'f2': f2, 'auc': auc}

        return scores


def prepareData(data):
    X_train_major, X_test_major, y_train_major, y_test_major = split(data['X_major'], data['y_major'], test_size=0.5)
    X_train_minor, X_test_minor, y_train_minor, y_test_minor = split(data['X_minor'], data['y_minor'], test_size=0.5)
    X_train = np.append(X_train_major, X_train_minor, axis=0).astype('float64')
    y_train = np.append(y_train_major, y_train_minor).astype('float64')
    X_test = np.append(X_test_major, X_test_minor, axis=0).astype('float64')
    y_test = np.append(y_test_major, y_test_minor).astype('float64')

    return X_train, y_train, X_test, y_test, {'majority': y_train_major[0], 'minority': y_train_minor[0]}



if __name__ == "__main__":
    main()
