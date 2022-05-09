# -*- coding: utf-8 -*-

from transformers import BertTokenizerFast, BertForSequenceClassification
import torch

from datasets import load_dataset

def main():
	model_name = 'bert-base-uncased'

	### dictionary: dataset name -> textual column
	ds_names = {
		'tweets_hate_speech_detection': 'tweet',
		'hate_speech18': 'text'
	}

	for ds_name in ds_names:
		print(ds_name)
		dataset = load_dataset(ds_name, split='train')
		target_dataset = dataset.filter(lambda example: example['label'] in [0, 1])

		#print(dataset.shape)
		#print(target_dataset.shape)
		#print(target_dataset.column_names)
		#print(target_dataset.features['label'].names)

		print(dir(target_dataset))


def examine(model_name, dataset, target_text):
	tokenizer = BertTokenizerFast.from_pretrained(model_name)
	model = BertForSequenceClassification.from_pretrained(model_name)

	dataset = dataset.map(lambda t:
		tokenizer(t[target_text], trancation=True, padding='max_length'),
		batched=True)
	dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

if __name__ == "__main__":
    main()
