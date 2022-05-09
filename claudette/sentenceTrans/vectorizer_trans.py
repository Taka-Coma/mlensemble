# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#os.environ['TRANSFORMERS_OFFLINE'] = '1'

from sentence_transformers import SentenceTransformer, models, InputExample, losses, SentencesDataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler 

import pickle
import json
import numpy as np


mode = 'triplet'
#mode = 'contrastive'

def main():
	models = [
		#'bert-base-uncased', 
		'nlpaueb/legal-bert-base-uncased',
	]

	fnames = [
		#'./vectors/triplet_bert-base-uncased.dump',
		'./vectors/triplet_legal-bert-base-uncased.dump',
	]

	#fnames = [
	#	#'./vectors/bert-base-uncased.dump',
	#	'./vectors/legal-bert-base-uncased.dump',
	#]


	for model, fname in zip(models, fnames):
		datasets = []
		for i in range(50):
			print(i)
			dataset = vectorize(i, model)
			datasets.append(dataset)

		with open(fname, 'wb') as w:
			pickle.dump(datasets, w)


vectorize_first = True
vector_map = None

def vectorize(i, model_name):
	global vectorize_first
	global vector_map

	with open(f'../dataset/{i}/train.json', 'r') as r:
		data = json.load(r)
		labels = data['labels']

		model = SentenceTransformer(model_name, device='cpu')

		base_model = models.Transformer(model_name)
		pooling_model = models.Pooling(base_model.get_word_embedding_dimension())
		dense_model = models.Dense(
		        in_features=pooling_model.get_sentence_embedding_dimension(),
		        out_features=base_model.get_word_embedding_dimension(),
		        activation_function=nn.Tanh())

		model = SentenceTransformer(
		        modules=[base_model,
		        pooling_model,
		        dense_model])

		if mode == 'triplet':
		        train_examples = [
			        InputExample(texts=[data['text'][i]], label=0) if data['labels'] == 0
			        else InputExample(texts=[data['text'][i]], label=1)
			            for i in range(len(data['labels']))
			        ]

		else:
			train_examples = [
			        InputExample(texts=[data['text'][i] for i in range(len(data['labels'])) if data['labels'] == 0], label=0), 
			        InputExample(texts=[data['text'][i] for i in range(len(data['labels'])) if data['labels'] == 1], label=1)
			        ]

		train_dataset = SentencesDataset(train_examples, model)

		train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
		#loss = losses.BatchHardSoftMarginTripletLoss(model)
		loss = losses.BatchAllTripletLoss(model, margin=20)
		#loss = losses.ContrastiveLoss(model)
		model.fit(train_objectives=[(train_dataloader, loss)], epochs=5, warmup_steps=100)

		X = None
		for txt in data['text']:
		    if X is None:
		        X = model.encode(txt)
		    else:
		        X = np.vstack((X, model.encode(txt)))
		train_data = {'X': X, 'labels': labels}

		### vectorizing without metric learning
		#if vectorize_first:
		#	X = model.encode(data['text'])
		#	vector_map = {txt: emb for txt, emb in zip(data['text'], X)}

		#else:
		#	X = None
		#	for txt in data['text']:
		#		if X is None:
		#			X = vector_map[txt]
		#		else:
		#			X = np.vstack((X, vector_map[txt]))

		#train_data = {'X': X, 'labels': labels}


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

			### vectorizing without metric learning
			#if vectorize_first:
			#	X = model.encode(data['text'])
			#	vector_map.update({
			#		txt: emb
			#		for txt, emb in zip(data['text'], X)
			#	})
			#	vectorize_first = False

			#else:
			#	X = None
			#	for txt in data['text']:
			#		if X is None:
			#			X = vector_map[txt]
			#		else:
			#			X = np.vstack((X, vector_map[txt]))
			#	
			#test_data = {'X': X, 'labels': labels}

		out = {'train': train_data, 'test': test_data}

	return out

if __name__ == "__main__":
    main()
