# -*- coding: utf-8 -*-

import json
from sklearn.model_selection import train_test_split

from os import makedirs

def main():
	with open('../data/tweets_hate_speech_detection.json', 'r') as r:
		data = json.load(r)

	x = data['text']
	y = data['labels']

	for i in range(10):
		makedirs(f'./dataset/{i}', exist_ok=True)
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)

		with open(f'./dataset/{i}/train.json', 'w') as w:
			json.dump({'text': x_train, 'labels': y_train}, w)

		with open(f'./dataset/{i}/test.json', 'w') as w:
			json.dump({'text': x_test, 'labels': y_test}, w)



if __name__ == "__main__":
    main()
