# MLEnsemble

| MLBagging | MLBoosting |
:--:|:--:
| ![mlbagging](https://user-images.githubusercontent.com/24326273/167428926-4f2eab61-070a-4ad1-b740-fa09f87b4b69.png) | ![mlboosting](https://user-images.githubusercontent.com/24326273/167429014-f41a7251-4aa0-4302-b9e4-416ebec648a3.png) |
| `mlbagging.py` | `mlboosting.py` |

| MLStacking | MLBoostacking |
:--:|:--:
| ![mlstacking](https://user-images.githubusercontent.com/24326273/167429038-50d5e7c0-0834-4b20-9f36-7bd61299d460.png) | ![mlboostacking](https://user-images.githubusercontent.com/24326273/167429063-e23adaaf-d2ed-4646-9bc6-68e71ed3a4f3.png) |
| `mlstacking.py` | `mlboostacking.py` |


# How to use

## Dataset
- claudette: http://claudette.eui.eu/ToS.zip
- hate-speech18: https://huggingface.co/datasets/hate_speech18 
- tweets-hate-speech-detection dataset: https://huggingface.co/datasets/tweets_hate_speech_detection

### Stats
| Dataset | #positives | #negatives | Imbalance Ratio |
:---|---:|---:|---:
| claudette | 1,032 | 8,382| 8.12 |
| hate-speech18 | 1,914 | 15,210 | 7.95 |
| tweets-hate-speech-detection | 2,242 | 29,720 | 13.26 |

### Preparing datasets
- Only for claudette: download the zip file from the above link, unzip it, and put the contents of the unzipped directories into `/datasets/claudette/original`
- Move to `/make_dataset` directory
- Execute `makeDataest_x.py` for each dataset x
	- `/datasets/x/i` directory is created for i-th dataset
	- The following files are in this directory
		- `train.json`
		- `test.json`

### Generating representations of texts
- Move to `/vectorizers` directory
- Execute `vectorizer.py` and `vectorizer_tripletloss.py`
	- `vectorizer.py`: generating representations from pretrained models
	- `vectorizer_tripletloss.py`: generating representations from pretrained models by using Triplet loss


## Baselines
- BERT (bert-large-uncased) : https://huggingface.co/bert-base-uncased
- LegalBERT (legal-bert-base-uncased) : https://huggingface.co/nlpaueb/legal-bert-base-uncased
- DeBERTa (deberta-v3-small-finetuned-hate_speech18) : https://huggingface.co/Narrativaai/deberta-v3-small-finetuned-hate_speech18
- DistilRoBERTa (distilroberta-finetuned-tweets-hate-speech) : https://huggingface.co/mrm8488/distilroberta-finetuned-tweets-hate-speech


## Executing classifiers

### Shallow
