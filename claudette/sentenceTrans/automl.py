# -*- coding: utf-8 -*-

### Loading data
import pickle
from scipy.sparse import vstack

### AutoML
from autosklearn.classification import AutoSklearnClassifier

### Evaluation
from sklearn.metrics import confusion_matrix as cm
from math import sqrt

### For saving results
import psycopg2 as psql
con = psql.connect('dbname=claudette_traditional')
cur = con.cursor()

### MIsc
import datetime


def main():
    path = './vectors/setencesTrans_paraphrase.dump'

    with open(path, 'rb') as r:
        dataset = pickle.load(r) 

    test(dataset, 'automl', emb='sentence_trans_para')

    cur.close()
    con.close()


def test(dataset, cls_name, emb=None):
    dbname = f'{cls_name}_{emb}'

    cur.execute(f'''
        create table if not exists {dbname} 
        (test_number int, metric text, value float8);
    ''')

    cur.execute(f'''
        create or replace view {dbname}_agg as select metric, avg(value), stddev(value)
        from {dbname} 
        group by metric
        order by metric;
    ''')

    con.commit()

    cur.execute(f'''
        select max(test_number)
        from {dbname}
    ''')
    if cur.fetchone()[0] == 49:
        print(f'skipped: {dbname}')
        return

    for i in range(len(dataset)):
        print(i)

        cur.execute(f'''
            select count(*)
            from {dbname}
            where test_number = %s
        ''', (i, ))
        if cur.fetchone()[0] > 0:
            print(f'skipped: {dbname} {i}')
            continue

        X_test, y_test = dataset[i]
        X_train, y_train = genTrain(dataset, i)

        model = AutoSklearnClassifier(n_jobs=-1, memory_limit=12000)

        model.fit(X_train, y_train)
        predicts = model.predict(X_test)

        accuracy, precision, recall, f1, f2, tpr, tnr, gmean = calc_scores(y_test, predicts)
        scores = {'gmean': gmean, 'tpr': tpr, 'tnr': tnr, 'accuracy': accuracy,
            'precision': precision, 'recall': recall, 'f1': f1, 'f2': f2}
 
        for metric in scores: 
            cur.execute(f'insert into {dbname} values (%s, %s, %s)',
                (i, metric, scores[metric]))

        con.commit()



def genTrain(dataset, except_ind):
    y = []
    X = None
    for i in range(len(dataset)):
        if i == except_ind:
            continue

        if X is None:
            X = dataset[i][0].copy()
        else:
            X = vstack((X, dataset[i][0]))

        y.extend(dataset[i][1])
    return X, y



def calc_scores(y_true, y_pred):
    tn, fp, fn, tp = cm(y_true, y_pred).ravel()
    if tn + fp > fn + tp:
        tpr = float(tp)/(fn+tp)
        tnr = float(tn)/(tn+fp) 
    else:
        tnr = float(tp)/(fn+tp)
        tpr = float(tn)/(tn+fp) 

    ## precision, recall, fscore
    if tp+fp == 0:
        precision = 0
    else:
        precision = float(tp) / (tp+fp)
    recall = tpr
    f1 = harmonicMean(precision, recall)
    f2 = harmonicMean(precision, recall, beta=2)

    accuracy = (tp+tn)/(tp+tn+fp+fn)

    return accuracy, precision, recall, f1, f2, tpr, tnr, sqrt(tpr * tnr)


def harmonicMean(f1, f2, beta=1):
    return (1+beta**2)*f1*f2 / ((beta**2) * f1 + f2)



if __name__ == "__main__":
    main()
