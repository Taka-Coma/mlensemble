# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer

import pickle
import json
import numpy as np


def main():
	models = [
		SentenceTransformer('bert-base-uncased', device='cpu'),
		SentenceTransformer('Narrativaai/deberta-v3-small-finetuned-hate_speech18', device='cpu'),
	]

	fnames = [
    	'./vectors/bert-base-uncased.dump',
    	'./vectors/deberta-v3-small-finetuned-hate_speech18.dump',
	]


	for model, fname in zip(models, fnames):
		vector_map = vectorizeAll(model)
		datasets = []
		for i in range(10):
			dataset = vectorize(i, vector_map)
			datasets.append(dataset)

		with open(fname, 'wb') as w:
			pickle.dump(datasets, w)


def vectorizeAll(model):
	with open('../../data/hate_speech18.json', 'r') as r:
		data = json.load(r)

	X = model.encode(data['text'])
	return {txt: x for txt, x in zip(data['text'], X)}


def vectorize(i, vector_map):
	with open(f'../dataset/{i}/train.json', 'r') as r:
		data = json.load(r)
		labels = data['labels']

		X = None
		for txt in data['text']:
			if X is None:
				X = vector_map[txt].copy()
			else:
				X = np.vstack((X, vector_map[txt]))

	train_data = {'X': X, 'labels': labels}

	with open(f'../dataset/{i}/test.json', 'r') as r:
		data = json.load(r)
		labels = data['labels']

		X = None
		for txt in data['text']:
			if X is None:
				X = vector_map[txt].copy()
			else:
				X = np.vstack((X, vector_map[txt]))
	test_data = {'X': X, 'labels': labels}

	out = {'train': train_data, 'test': test_data}

	return out

if __name__ == "__main__":
    main()
