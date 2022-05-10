# -*- coding: utf-8 -*-

import json
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from os import makedirs

def main():
	ds_name = 'hate_speech18'

	dataset = load_dataset(ds_name, split='train')
	data = dataset.to_dict()
	x = data['text'][:]
	y = data['label'][:]

	for i in range(10):
		makedirs(f'../datasets/{ds_name}/{i}', exist_ok=True)
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

		with open(f'../datasets/{ds_name}/{i}/train.json', 'w') as w:
			json.dump({'text': x_train, 'labels': y_train}, w)

		with open(f'../datasets/{ds_name}/{i}/test.json', 'w') as w:
			json.dump({'text': x_test, 'labels': y_test}, w)



if __name__ == "__main__":
    main()
