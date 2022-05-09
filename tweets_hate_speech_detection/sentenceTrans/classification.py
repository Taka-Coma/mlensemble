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

from mmensemble import MLEnsembleClassifier as MLE
from mlboost import MLBoost
from mladaboost import MLAdaBoost
#from mlstacking import MLStackingClassifier as MLS

### Preprocessing
from metric_learn import LMNN, SCML_Supervised as SCML

### Evaluation
from utils import PRF_TPR_TNR_gmean_AUC

### For saving results
import psycopg2 as psql
con = psql.connect(dbname='tweets_hate', user='taka-coma', host='192.168.111.200')
cur = con.cursor()

### Misc
import datetime


#classifiers = ['knn', 'rf', 'brf', 'ee', 'svmlin', 'svmrbf', 'mle', 'knn_lmnn', 'mle_knn', 'lr']
#classifiers = ['brf', 'ee']
#classifiers = ['mle', 'mlboost']
#classifiers = ['mlstacking']
#classifiers = ['ee', 'mle', 'mlboost', 'mlstacking']
classifiers = ['mladaboost']

def main():
	#paths =[ './vectors/setencesTrans_paraphrase.dump', './vectors/setencesTrans_stsb.dump']
	#embs = ['sentence_trans_para', 'sentence_trans_stsb'] 

	paths =[
            './vectors/bert-base-uncased.dump',
            './vectors/distilroberta-finetuned-tweets-hate-speech.dump'
            ]
	embs = [
            'bert_base_uncased',
            'distilroberta_finetuned_tweets_hate_speech'
            ]

	#paths =[
	#	'./vectors/triplet_bert-base-uncased.dump',
	#	'./vectors/triplet_distilroberta-finetuned-tweets-hate-speech.dump'
	#]
	#embs = [
	#	'triplet_bert_base_uncased',
	#	'triplet_distilroberta_finetuned_tweets_hate_speech'
	#]

	for path, emb in zip(paths, embs):
		test_vector(path, emb)

	cur.close()
	con.close()


def test_vector(path, emb):
	with open(path, 'rb') as r:
		dataset = pickle.load(r) 

	for cls in classifiers:
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

	for i in reversed(range(len(dataset))):
                print(i)

                cur.execute(f'''
						select count(*)
						from {dbname}
						where test_number = %s
					''', (i, ))
                if cur.fetchone()[0] > 0:
                    print(f'skipped: {dbname} {i}')
                    continue

                X_train = dataset[i]['train']['X']
                y_train = dataset[i]['train']['labels']
                X_test = dataset[i]['test']['X']
                y_test = dataset[i]['test']['labels']

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
                    cls = MLE(n_estimators=5, n_jobs=-1)
                        #X_train = X_train.toarray()

                elif cls_name == 'mle_knn':
                    cls = MLE(base_estimator=kNN(), n_jobs=-1)
                        #X_train = X_train.toarray()

                elif cls_name == 'mlboost':
                    cls = MLBoost()
                        #X_train = X_train.toarray()

                elif cls_name == 'mladaboost':
                    cls = MLAdaBoost()
                        #X_train = X_train.toarray()

                #elif cls_name == 'mlstacking':
                #    cls = MLS(n_estimators=5, n_jobs=-1)
                #        #X_train = X_train.toarray()


                cls.fit(X_train, y_train)
                predicts = cls.predict(X_test)
                predict_proba = cls.predict_proba(X_test)

                scores = PRF_TPR_TNR_gmean_AUC(y_test, predicts, predict_proba)

                for metric in scores: 
                    cur.execute(f'insert into {dbname} values (%s, %s, %s)',
                            (i, metric, scores[metric]))

                con.commit()

if __name__ == "__main__":
	main()
