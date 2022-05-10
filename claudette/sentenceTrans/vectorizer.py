# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer

import pickle
from glob import glob


def main():
    #model = SentenceTransformer('nlpaueb/legal-bert-base-uncased', device='cpu')
    model = SentenceTransformer('bert-base-uncased', device='cpu')

    labels = []
    for path in glob('../data/Labels/*.txt'):
        with open(path, 'r') as r:
            labels.append([line.strip() for line in r])

    Xs = []
    for path in glob('../data/Sentences/*.txt'):
        with open(path, 'r') as r:
            sentences = [line.strip() for line in r]
            X = model.encode(sentences)
        Xs.append(X)

    out = list(zip(Xs, labels))

    #with open('./vectors/legal-bert-base-uncased.dump', 'wb') as w:
    with open('./vectors/bert-base-uncased.dump', 'wb') as w:
        pickle.dump(out, w)

if __name__ == "__main__":
    main()
