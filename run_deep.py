# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_DISABLED'] = 'true'

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from math import sqrt
from datasets import Dataset

from collections import Counter

import json
from glob import glob


### For saving results
import psycopg2 as psql
con = psql.connect(dbname='claudette', user='taka-coma', host='192.168.111.200')
#con = psql.connect(dbname='hate_speech18', user='taka-coma', host='192.168.111.200')
#con = psql.connect(dbname='tweets_hate', user='taka-coma', host='192.168.111.200')
cur = con.cursor()

ds_name = 'claudette'
#ds_name = 'hate_speech18'
#ds_name = 'tweets_hate_speech_detection'


### BERT model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizerFast.from_pretrained(model_name)

### LegalBERT model
#model_name = 'nlpaueb/legal-bert-base-uncased'
#tokenizer = BertTokenizerFast.from_pretrained(model_name)

### DeBERTa model
#model_name = 'Narrativaai/deberta-v3-small-finetuned-hate_speech18'
#tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

### RoBERTa model
#model_name = 'mrm8488/distilroberta-finetuned-tweets-hate-speech'
#tokenizer = RobertaTokenizerFast.from_pretrained(model_name)


def main():
	if model_name.find('/') > -1:
		tblname = f'{model_name[model_name.find("/")+1:].replace("-", "_").replace(".", "")}'
	else:
		tblname = f'{model_name.replace("-", "_").replace(".", "")}'

	cur.execute(f'''
					create table if not exists {tblname} 
					(test_number int, metric text, value float8);
				''')

	cur.execute(f'''
					create or replace view {tblname}_agg as select metric, avg(value), stddev(value)
					from {tblname} 
					group by metric
					order by metric;
				''')

	con.commit()



	for i in range(10):
		print(i)

		cur.execute(f'''
			select count(*)
			from {tblname}
			where test_number = %s
		''', (i,))
		if cur.fetchone()[0] > 0:
			continue

		train_ds, test_ds = loadData(i)

		scores = examine(train_ds, test_ds, tblname, i)
		for metric in scores: 
			cur.execute(f'insert into {tblname} values (%s, %s, %s)',
						(i, metric, scores[metric]))

		con.commit()

	cur.close()
	con.close()


def examine(train_ds, test_ds, tblname, i):
	training_args = TrainingArguments(
		output_dir=f"test_trainer/{tblname}_{i}", 
		evaluation_strategy="epoch",
		save_strategy="epoch",
		num_train_epochs=10,
		per_device_train_batch_size=16,
		load_best_model_at_end=True,
	)

	### for BERT and LegalBERT
	model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

	### for DeBERTa
	#model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=4)

	### for RoBERTa
	#model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_ds,
		eval_dataset=test_ds,
		compute_metrics=compute_metrics,
		data_collator=DataCollatorWithPadding(tokenizer),
		tokenizer=tokenizer,
	)

	trainer.train()
	metrics = trainer.evaluate()

	return {
		'precision': metrics['eval_precision'],
		'recall': metrics['eval_recall'],
		'gmean': metrics['eval_gmean'],
		'f1': metrics['eval_f1'],
		'f2': metrics['eval_f2'],
	}




def compute_metrics(eval_pred):
	logits, labels = eval_pred
	preds = np.argmax(logits, axis=-1) 

	acc = accuracy_score(labels, preds)
	pre, rec, f2, sup = precision_recall_fscore_support(labels, preds, beta=2)
	f1 = 2*pre[1]*rec[1] / (pre[1] + rec[1])
	out =  {
		'accuracy': acc,
		'precision': pre[1],
		'recall': rec[1],
		'gmean': sqrt(rec[0]*rec[1]),
		'f1': f1,
		'f2': f2[1]
	}

	return out


def loadData(i):
	path_head = f'./dataset/{ds_name}/{i}'

	train_dataset = loadSplit(f'{path_head}/train.json')
	test_dataset = loadSplit(f'{path_head}/test.json')

	return train_dataset, test_dataset


def loadSplit(path):
	with open(path, 'r') as r:
		data = json.load(r)
	ds = Dataset.from_dict(data)
	ds = ds.map(tokenize_fn, batched=True)
	return ds


def tokenize_fn(examples):
	return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def readSentences(path):
	return [open(txt_file, 'r').read() for txt_file in glob(f'{path}/*')]


if __name__ == "__main__":
    main()
