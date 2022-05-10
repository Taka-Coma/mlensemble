# -*- coding: utf-8 -*-

from glob import glob
from os import makedirs
import json

def main():
	data_root = '../datasets/claudette'
	sentence_path_head = f'{data_root}/original/Sentences'
	label_path_head = f'{data_root}/original/Labels'

	data = {}
	for path in glob(f'{sentence_path_head}/*'):
		fname = path[path.rfind('/')+1:]

		with open(f'{sentence_path_head}/{fname}', 'r') as r:
			sentences = [line.strip() for line in r]
		with open(f'{label_path_head}/{fname}', 'r') as r:
			labels = [0 if line.strip() != '1' else 1 for line in r]

		data[fname] = {
			'sentences': sentences,
			'labels': labels
		}


	for i, fname in enumerate(data.keys()):
		save_dir = f'../datasets/claudette/{i}'
		makedirs(save_dir, exist_ok=True)

		test_data = {
			'text': data[fname]['sentences'],
			'labels': data[fname]['labels']
		}

		train_data = genTrain(data, fname)
	
		with open(f'{save_dir}/train.json', 'w') as w:
			json.dump(train_data, w)

		with open(f'{save_dir}/test.json', 'w') as w:
			json.dump(test_data, w)


def genTrain(data, except_name):
	X = []
	labels = []

	for fname in data:
		if fname == except_name:
			continue

		X = X + data[fname]['sentences']
		labels = labels + data[fname]['labels']

	return {'text': X, 'labels': labels}


	


if __name__ == "__main__":
    main()
