# -*- coding: utf-8 -*-

import csv
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split as split

from os.path import exists
import pickle


def examineInTrainingData(classifier, X, labels):
    ### prepare data for experiment
    X_train, X_test, y_train, y_test = split(X, labels)

    ### classification
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    f1 = f1_score(y_test, y_pred, average=None)
    print('individual:', f1)

    f1 = f1_score(y_test, y_pred, average='macro')
    print('macro:', f1)




def convert2Vectors(sentences, vector_path):
    if exists(vector_path):
        with open(vector_path, 'rb') as r:
            X = pickle.load(r)
    else:
        model = SentenceTransformer('paraphrase-distilroberta-base-v1', device='cpu')
        X = model.encode(sentences)
        with open(vector_path, 'wb') as w:
            pickle.dump(X, w)
    return X


def readTestData():
    with open('../data/SDP_test.csv', 'r') as r:
        reader = csv.reader(r)

        ### remove header
        next(reader)

        sentences, ids = zip(*[(r[-1], r[0]) for r in reader])
        vector_path = './vectors/sentences_test_paraphrase_distilroberta.dump'
        X = convert2Vectors(sentences, vector_path)
        return X, ids 



def readTrainingData():
    with open('../data/SDP_train.csv', 'r') as r:
        reader = csv.reader(r)

        ### remove header
        next(reader)

        sentences, labels = zip(*[(r[-2], int(r[-1])) for r in reader])
        vector_path = './vectors/sentences_train_paraphrase_distilroberta.dump'
        X = convert2Vectors(sentences, vector_path)

        return X, labels





#def readTestData():
#    with open('../data/SDP_test.csv', 'r') as r:
#        reader = csv.reader(r)
#
#        ### remove header
#        next(reader)
#
#        sentences, ids = zip(*[(r[-1], r[0]) for r in reader])
#        return sentences, ids 
#
#
#
#def readTrainingData():
#    with open('../data/SDP_train.csv', 'r') as r:
#        reader = csv.reader(r)
#
#        ### remove header
#        next(reader)
#
#        sentences, labels = zip(*[(r[-2], int(r[-1])) for r in reader])
#        return sentences, labels
