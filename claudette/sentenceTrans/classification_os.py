# -*- coding: utf-8 -*-

### Loading data
import pickle
from scipy.sparse import vstack

### Classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from imblearn.ensemble import BalancedRandomForestClassifier as BRF, EasyEnsembleClassifier as EE

### Preprocessing 
from smote_variants import SMOTE, ProWSyn, polynom_fit_SMOTE as PFSMOTE

### Evaluation
from utils import PRF_TPR_TNR_gmean_AUC

### For saving results
import psycopg2 as psql
con = psql.connect('dbname=claudette')
cur = con.cursor()

### MIsc
import datetime
#from multiprocessing import Pool


def main():
	oss = ['SMOTE', 'ProWSyn', 'PFSMOTE']

    #paths = ['./vectors/setencesTrans_paraphrase.dump', './vectors/setencesTrans_stsb.dump']
    #embs = ['sentence_trans_para', 'sentence_trans_stsb']

	paths =[ './vectors/bert-base-uncased.dump', './vectors/legal-bert-base-uncased.dump', './vectors/custom-legalbert.dump']
	embs = ['bert_base_uncased', 'legal_bert_base_uncased', 'custom_legalbert'] 

	for os in oss:
		for path, emb in zip(paths, embs):
			embedding_approach(os, path, emb)

	cur.close()
	con.close()


def embedding_approach(os, path, emb):
    with open(path, 'rb') as r:
        dataset = pickle.load(r) 

    #for cls in ['knn', 'rf', 'svmlin', 'lr']:
    for cls in ['rf']:
        print(f'{datetime.datetime.now()} || {cls}')

        test(dataset, cls, emb=emb, os_name=os)


def test(dataset, cls_name, emb=None, os_name='SMOTE'):
    dbname = f'{cls_name}_{emb}_{os_name}'

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
        y_test = [0 if v == '-1' else int(v) for v in y_test]
        X_train, y_train = genTrain(dataset, i)
        X_train = X_train.toarray()
        #X_test = X_test.toarray()

        if os_name == 'SMOTE':
            sampler = SMOTE(n_jobs=-1)
        elif os_name == 'ProWSyn':
            sampler = ProWSyn(n_jobs=-1)
        elif os_name == 'PFSMOTE':
            sampler = PFSMOTE()

        X_train, y_train = sampler.sample(X_train, y_train)

        if cls_name == 'rf':
            cls = RF(n_jobs=-1)
        elif cls_name == 'brf':
            cls = BRF(n_jobs=-1)
        elif cls_name == 'ee':
            cls = EE(n_jobs=-1)
        elif cls_name == 'lr':
            cls = LR()
        elif cls_name == 'svmlin':
            cls = LinearSVC()
        elif cls_name == 'svmrbf':
            cls = SVC(kernel='rbf')
        elif cls_name == 'knn':
            cls = KNN(n_jobs=-1)

        model = cls
        model.fit(X_train, y_train)
        predicts = model.predict(X_test)

        scores = PRF_TPR_TNR_gmean_AUC(y_test, predicts)
 
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




if __name__ == "__main__":
    main()
