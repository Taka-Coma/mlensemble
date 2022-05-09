# -*- coding: utf-8 -*-

### Loading data
import pickle
from scipy.sparse import vstack

### Classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as kNN
from imblearn.ensemble import BalancedRandomForestClassifier as BRF, EasyEnsembleClassifier as EE
from imblearn.ensemble import BalancedBaggingClassifier as BBC
from mmensemble import MLEnsembleClassifier as MLE, MMEnsembleClassifier as MME

### Preprocessing
from metric_learn import LMNN, SCML_Supervised as SCML

### AutoML
from autosklearn.classification import AutoSklearnClassifier

### Evaluation
from sklearn.metrics import confusion_matrix as cm
from math import sqrt

### For saving results
import psycopg2 as psql
con = psql.connect('dbname=claudette')
cur = con.cursor()

### Misc
import datetime



def main():
	paths =[ './vectors/setencesTrans_paraphrase.dump', './vectors/setencesTrans_stsb.dump']
	embs = ['sentence_trans_para', 'sentence_trans_stsb'] 

	for path, emb in zip(paths, embs):
		test_vector(path, emb)

	cur.close()
	con.close()


def test_vector(path, emb):
    with open(path, 'rb') as r:
        dataset = pickle.load(r) 

    for cls in ['knn', 'rf', 'brf', 'ee', 'svmlin', 'svmrbf', 'mle', 'knn_lmnn', 'mle_knn', 'lr']:
        print(f'{datetime.datetime.now()} || {cls}')

        test(dataset, cls, emb=emb)


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

        if cls_name == 'rf':
            cls = RF(n_jobs=-1)
        elif cls_name == 'brf':
            cls = BRF(n_jobs=-1)
        elif cls_name == 'ee':
            cls = EE(n_jobs=-1)
        elif cls_name == 'svmlin':
            cls = LinearSVC()
        elif cls_name == 'lr':
            cls = LR()
        elif cls_name == 'svmrbf':
            cls = SVC(kernel='rbf')
        elif cls_name == 'knn':
            cls = kNN(n_jobs=-1)

        elif cls_name == 'knn_lmnn':
            cls = Pipeline([
                ('metric learner', LMNN()),
                ('classifier', kNN(n_jobs=-1))
            ])
            X_train = X_train.toarray()

        elif cls_name == 'svmlin_lmnn':
            cls = Pipeline([
                ('metric learner', LMNN()),
                ('classifier', LinearSVC())
            ])
            X_train = X_train.toarray()

        elif cls_name == 'mle':
            cls = MLE(n_jobs=-1)
            X_train = X_train.toarray()

        elif cls_name == 'mle_knn':
            cls = MLE(base_estimator=kNN(), n_jobs=-1)
            X_train = X_train.toarray()

        elif cls_name == 'mme':
            cls = MME(n_jobs=-1)
            X_train = X_train.toarray()

        elif cls_name == 'mme_knn':
            cls = MME(base_estimator=kNN(), n_jobs=-1)
            X_train = X_train.toarray()



        cls.fit(X_train, y_train)
        predicts = cls.predict(X_test)

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
    if f1 == 0 and f2 == 0:
        return 0
    else:
        return (1+beta**2)*f1*f2 / ((beta**2) * f1 + f2)



if __name__ == "__main__":
    main()
