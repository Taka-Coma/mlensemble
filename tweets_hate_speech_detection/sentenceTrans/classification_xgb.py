# -*- coding: utf-8 -*-

### Loading data
import pickle
from scipy.sparse import vstack

### Classification
import xgboost as xgb

### Preprocessing
from metric_learn import LMNN, SCML_Supervised as SCML

### Evaluation
import numpy as np
from classification import calc_scores

### For saving results
import psycopg2 as psql
con = psql.connect('dbname=claudette')
cur = con.cursor()

### Misc
from collections import Counter
import datetime


def main():
    path = './vectors/setencesTrans_paraphrase.dump'
    #path = './vectors/setencesTrans_stsb.dump'

    with open(path, 'rb') as r:
        dataset = pickle.load(r) 

    cls = 'xgb'

    for weight in [False, True]:
            ml = False
        #for ml in [False, True]:
            print(f'{datetime.datetime.now()} || {cls}')

            test(dataset, cls, emb='sentence_trans_para', weight_flag=weight, ml=ml)
            #test(dataset, cls, emb='sentence_trans_stsb', weight_flag=weight, ml=ml)

    cur.close()
    con.close()


def test(dataset, cls_name, emb=None, weight_flag=None, ml=False):
    dbname = f'{cls_name}_{emb}'

    if weight_flag:
        dbname = f'{dbname}_weight'

    if ml:
        dbname = f'{dbname}_ml'

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
        print(f'{datetime.datetime.now()} || {dbname} {i}')

        cur.execute(f'''
            select count(*)
            from {dbname}
            where test_number = %s
        ''', (i, ))
        if cur.fetchone()[0] > 0:
            print(f'skipped: {dbname} {i}')
            continue

        X_test, y_test = dataset[i]
        y_test = [0 if v == '-1' else int(v) for v in y_test]
        X_train, y_train = genTrain(dataset, i)

        if ml:
            X_train = X_train.toarray()

        counter = Counter(y_train).most_common(2)
        major = counter[0][1]
        minor = counter[1][1]
        weight = major/minor

        ### prepare data
        if ml:
            metric_learner = LMNN()
            #metric_learner = SCML(n_basis=300)
            X_train = metric_learner.fit_transform(X_train, y_train)
            X_test = metric_learner.transform(X_test)
            dtrain = xgb.DMatrix(X_train, label=y_train)

        elif emb is not None:
            dtrain = xgb.DMatrix(X_train.tocsr(), label=y_train)
        else:
            dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test)

        params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
            }
        if weight_flag:
            params['scale_pos_weight'] = weight

        model = xgb.train(params, dtrain, 20)
        predicts = np.round(model.predict(dtest))

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

    y_out = [0 if v == '-1' else int(v) for v in y]
    return X, y_out


def harmonicMean(f1, f2, beta=1):
    return (1+beta**2)*f1*f2 / ((beta**2) * f1 + f2)



if __name__ == "__main__":
    main()
