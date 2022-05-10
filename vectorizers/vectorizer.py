# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from sentence_transformers import SentenceTransformer, models

import pickle
import json
import numpy as np


def main():
	### for claudette
	models = [
		'bert-base-uncased', 
		'nlpaueb/legal-bert-base-uncased',
	]
	fnames = [
		'./vectors/claudette/bert-base-uncased.dump',
		'./vectors/claudette/legal-bert-base-uncased.dump',
	]

	### for hate-speech18
	#models = [
	#	'bert-base-uncased',
	#	'Narrativaai/deberta-v3-small-finetuned-hate_speech18',
	#]
	#fnames = [
    #	'./vectors/hate_speech18/bert-base-uncased.dump',
    #	'./vectors/hate_speech18/deberta-v3-small-finetuned-hate_speech18.dump',
	#]

	### tweets-hate-speech
	#models = [
	#	'bert-base-uncased',
	#	'mrm8488/distilroberta-finetuned-tweets-hate-speech',
	#]
	#fnames = [
    #	'./vectors/tweet_hate/bert-base-uncased.dump',
    #	'./vectors/tweet_hate/distilroberta-finetuned-tweets-hate-speech.dump',
	#]

	for model, fname in zip(models, fnames):
		datasets = []
		for i in range(50):
			print(i)
			dataset = vectorize(i, model)
			datasets.append(dataset)

		with open(fname, 'wb') as w:
			pickle.dump(datasets, w)


def vectorize(i, model_name):
	with open(f'../dataset/{i}/train.json', 'r') as r:
		data = json.load(r)
		labels = data['labels']

		model = SentenceTransformer(model_name, device='cpu')

		X = None
		for txt in data['text']:
		    if X is None:
		        X = model.encode(txt)
		    else:
		        X = np.vstack((X, model.encode(txt)))
		train_data = {'X': X, 'labels': labels}

		with open(f'../dataset/{i}/test.json', 'r') as r:
			data = json.load(r)
			labels = data['labels']

			X = None
			for txt in data['text']:
				if X is None:
					X = model.encode(txt)
				else:
					X = np.vstack((X, model.encode(txt)))
			test_data = {'X': X, 'labels': labels}

		out = {'train': train_data, 'test': test_data}

	return out

if __name__ == "__main__":
    main()
