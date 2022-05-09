# MLEnsemble

| MLBagging | MLBoosting | MLStacking | MLBoostacking |
---|---|---|---
| ![mlbagging](https://user-images.githubusercontent.com/24326273/167428926-4f2eab61-070a-4ad1-b740-fa09f87b4b69.png) | ![mlboosting](https://user-images.githubusercontent.com/24326273/167429014-f41a7251-4aa0-4302-b9e4-416ebec648a3.png) | ![mlstacking](https://user-images.githubusercontent.com/24326273/167429038-50d5e7c0-0834-4b20-9f36-7bd61299d460.png) | ![mlboostacking](https://user-images.githubusercontent.com/24326273/167429063-e23adaaf-d2ed-4646-9bc6-68e71ed3a4f3.png) |



## MLBagging
![mlbagging](https://user-images.githubusercontent.com/24326273/167428926-4f2eab61-070a-4ad1-b740-fa09f87b4b69.png)

## MLBoosting
![mlboosting](https://user-images.githubusercontent.com/24326273/167429014-f41a7251-4aa0-4302-b9e4-416ebec648a3.png)

## MLStacking
![mlstacking](https://user-images.githubusercontent.com/24326273/167429038-50d5e7c0-0834-4b20-9f36-7bd61299d460.png)

## MLBoostacking
![mlboostacking](https://user-images.githubusercontent.com/24326273/167429063-e23adaaf-d2ed-4646-9bc6-68e71ed3a4f3.png)


# Experiment

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


## Baselines
- BERT (bert-large-uncased) : https://huggingface.co/bert-base-uncased
- legal-bert-base-uncased : https://huggingface.co/nlpaueb/legal-bert-base-uncased
- deberta-v3-small-finetuned-hate_speech18 : https://huggingface.co/Narrativaai/deberta-v3-small-finetuned-hate_speech18
- distilroberta-finetuned-tweets-hate-speech : https://huggingface.co/mrm8488/distilroberta-finetuned-tweets-hate-speech
