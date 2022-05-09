# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['WANDB_DISABLED'] = 'true'

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from math import sqrt
from datasets import Dataset

from torch.utils.data import WeightedRandomSampler
from collections import Counter

from glob import glob


### For saving results
import psycopg2 as psql
con = psql.connect('dbname=claudette')
cur = con.cursor()


### BERT model
#model_name = 'bert-base-uncased'
#model_name = 'nlpaueb/legal-bert-base-uncased'
model_name = 'zlucia/custom-legalbert'
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def main():

	#if model_name.find('/') > -1:
	#	dbname = f'{model_name[model_name.find("/")+1:].replace("-", "_").replace(".", "")}_us'
	#else:
	#	dbname = f'{model_name.replace("-", "_").replace(".", "")}_us'

	if model_name.find('/') > -1:
		dbname = f'{model_name[model_name.find("/")+1:].replace("-", "_").replace(".", "")}_us_weightCE'
	else:
		dbname = f'{model_name.replace("-", "_").replace(".", "")}_us_weightCE'

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



	for i in range(50):
		print(i)

		cur.execute(f'''
			select count(*)
			from {dbname}
			where test_number = %s
		''', (i,))
		if cur.fetchone()[0] > 0:
			continue

		train_ds, test_ds = loadData(i)

		scores = examine(train_ds, test_ds, dbname)
		for metric in scores: 
			cur.execute(f'insert into {dbname} values (%s, %s, %s)',
						(i, metric, scores[metric]))

		con.commit()

	cur.close()
	con.close()


def examine(train_ds, test_ds, dbname):
	training_args = TrainingArguments(
		output_dir=f"test_trainer/{dbname}", 
		evaluation_strategy="epoch",
		save_strategy="epoch",
		num_train_epochs=10,
		per_device_train_batch_size=16,
		load_best_model_at_end=True,
	)

	trainer = BalancedTrainer(
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

class BalancedTrainer(Trainer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.data_dict = self.train_dataset.to_dict()
		freq = Counter(self.data_dict['labels']).most_common(2)
		self.class_count = np.array([freq[0][1], freq[1][1]])

	def _get_train_sampler(self):
		weight = 1./self.class_count
		sample_weight = torch.from_numpy(np.array([weight[t] for t in self.data_dict['labels']]))

		return WeightedRandomSampler(
			sample_weight, 
			len(sample_weight)
		)

	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.get('labels')
		outputs = model(**inputs)
		logits = outputs.get("logits")

		weights = 1./self.class_count
		weights = weights / weights.sum()
		weights = torch.from_numpy(weights).float().to('cuda')

		loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
		loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))

		return (loss, outputs) if return_outputs else loss


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
	path_head = f'../dataset/{i}'
	class_names = ['fair', 'unfair']

	train_dataset = loadSplit(f'{path_head}/train')
	test_dataset = loadSplit(f'{path_head}/test')

	return train_dataset, test_dataset


def loadSplit(path):
	neg_sents = readSentences(f'{path}/fair')
	pos_sents = readSentences(f'{path}/unfair')

	labels = [0 for _ in range(len(neg_sents))] + [1 for _ in range(len(pos_sents))]
	sents = neg_sents + pos_sents

	#ds = Dataset.from_dict({'text': sents, 'label': labels})
	ds = Dataset.from_dict({'text': sents, 'labels': labels})
	ds = ds.map(tokenize_fn, batched=True)
	return ds


def tokenize_fn(examples):
	return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
	#return tokenizer(examples['text'], padding='max_length', truncation=True)

def readSentences(path):
	return [open(txt_file, 'r').read() for txt_file in glob(f'{path}/*')]


if __name__ == "__main__":
    main()
