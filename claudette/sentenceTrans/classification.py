# -*- coding: utf-8 -*-

### Loading data
import pickle
from scipy.sparse import vstack

### Classification
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as kNN
from imblearn.ensemble import EasyEnsembleClassifier as EE
from imblearn.ensemble import RUSBoostClassifier

### mlbagging
from mlbagging import MLBaggingClassifier
### mlboostacking
from mlboostacking import MLBoostackingClassifier
### mlboosting
from mlboosting import MLBoostingClassifier
### mlstacking
from mlstacking import MLStackingClassifier 

### Preprocessing
from metric_learn import LMNN

### Evaluation
from utils import PRF_TPR_TNR_gmean_AUC

### For saving results
import psycopg2 as psql
con = psql.connect(dbname='claudette', user='taka-coma', host='192.168.111.200')
cur = con.cursor()

### Misc
import datetime


classifiers = ['ee', 'rusboost', 'mlbagging', 'mlboost', 'mlstacking', 'mladaboost']

def main():
	paths =[ 
			'./vectors/bert-base-uncased.dump',
			'./vectors/legal-bert-base-uncased.dump',
            './vectors/triplet_bert-base-uncased.dump',
            './vectors/triplet_legal-bert-base-uncased.dump'
            ]
    embs = [
			'bert_base_uncased',
			'legal_bert_base_uncased',
            'triplet_bert_base_uncased',
            'triplet_legal_bert_base_uncased',
			]

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

		X_train = dataset[i]['train']['X']
		y_train = dataset[i]['train']['labels']
		X_test = dataset[i]['test']['X']
		y_test = dataset[i]['test']['labels']

		if cls_name == 'ee':
			cls = EE(n_jobs=-1)

		elif cls_name == 'rusboost':
			cls = RUSBoostClassifier()

		elif cls_name == 'mlbagging':
			cls = MLBaggingClassifier(n_features=X_train.shape[1])

		elif cls_name == 'mlboosting':
			cls = MLBoostingClassifier()

		elif cls_name == 'mlstacking':
			cls = MLStackingClassifier()

		elif cls_name == 'mlboostacking':
			cls = MLBoostackingClassifier()


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
