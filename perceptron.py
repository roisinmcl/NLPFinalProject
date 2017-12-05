#!/usr/bin/env python3
"""
Starter code for our Political Ideology Classifier

Usage: python perceptron.py NITERATIONS

(Adapted from Alan Ritter)

"""
import sys, os, glob, time

from collections import Counter
from math import log
from numpy import mean
import numpy as np

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

from evaluation import Eval
from sentiment import NBClassifier, get_sentiment


SENTIMENT_WORDS = ['taxes', 'tax', 'gun control', 'immigration', 'obamacare']

# TODO: global variables are bad, fix later
global sentiment_classifier

# taken from https://stackoverflow.com/questions/13214809/pretty-print-2d-python-list
def print_matrix(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def load_docs(direc, lemmatize):
    """Return a list of word-token-lists, one per document.
    Words are optionally lemmatized with WordNet."""

    labelMap = {}   # docID => gold label, loaded from mapping file
    for file_path in glob.glob(os.path.join(direc, '*.txt')):
        filename = os.path.basename(file_path)
        party = filename[-7]
        labelMap[filename] = party 

    # create parallel lists of documents and labels
    docs, labels = [], []
    for file_path in glob.glob(os.path.join(direc, '*.txt')):
        filename = os.path.basename(file_path)
        doc_words = []
        with open(os.path.join(direc, filename)) as inF:
            for ln in inF:
                if lemmatize:
                    words = word_tokenize(ln)
                    tagged_words = pos_tag(words)
                    doc_words = []
                    for word in tagged_words:
                        if get_pos_tag(word[1]) == None:
                            lemmatized_word = WordNetLemmatizer().lemmatize(word[0])
                        else:
                            lemmatized_word = WordNetLemmatizer().lemmatize(word[0], get_pos_tag(word[1]))
                        doc_words.append(lemmatized_word)
                else:
                    doc_words += ln.split(' ')
        
        docs.append(doc_words)
        label = labelMap[filename]
        labels.append(label)

    return docs, labels

def extract_feats(doc, lemmatized_docs=None):
    """
    Extract input features (percepts) for a given document.
    Each percept is a pairing of a name and a boolean, integer, or float value.
    A document's percepts are the same regardless of the label considered.
    """
    global sentiment_classifier

    ff = Counter()

    # Unigram Features
    ff += Counter(doc)
    ff['bias'] = 1
    

    # Bigram Features
    offset_doc = doc[1:]
    bigrams = zip(doc, offset_doc)
    ff += Counter(bigrams)
    ff['bias'] = 1
    

    # Lemmatized Features - not used in most accurate model
    ff += Counter(lemmatized_docs) 
    ff['bias'] = 1


    # Case Normalization Features
    lowercase_doc = []
    for word in doc:
        lowercase_doc.append(word.lower())
    ff += Counter(lowercase_doc)
    ff['bias'] = 1

    # Senitment Analysis
    text = ' '.join(lowercase_doc)
    for word in SENTIMENT_WORDS:
        key = "sent-" + word
        if word in lowercase_doc:
            sentiment = sentiment_classifier.get_sentiment(text)
#            ff[key] = sentiment
#            sentiment = get_sentiment(text)
        else:
            ff[key] = 0
        #print("Sentiment for word " + word)
        #print(ff[word])

    return ff

def load_featurized_docs(datasplit):
    rawdocs, labels = load_docs(datasplit, lemmatize=False)
    
    #lemmatized_docs, lemma_labels = load_docs(datasplit, lemmatize=True)
    
    assert len(rawdocs)==len(labels)>0,datasplit
    featdocs = []
    for d in range(0, len(rawdocs)):
        # Use second call to extract_feats if using lemmatizing feature, first call otherwise
        featdocs.append(extract_feats(rawdocs[d]))
        #featdocs.append(extract_feats(rawdocs[d], lemmatized_docs[d]))
    return featdocs, labels

class Perceptron:
    def __init__(self, train_docs, train_labels, MAX_ITERATIONS=100, dev_docs=None, dev_labels=None):
        self.CLASSES = ['D', 'R', 'I']
        self.MAX_ITERATIONS = MAX_ITERATIONS
        self.dev_docs = dev_docs
        self.dev_labels = dev_labels
        self.weights = {l: Counter() for l in self.CLASSES}
        self.learn(train_docs, train_labels)

    def copy_weights(self):
        """
        Returns a copy of self.weights.
        """
        return {l: Counter(c) for l,c in self.weights.items()}

    def score(self, doc, label):
        """
        Returns the current model's score of labeling the given document
        with the given label.
        """        
        doc_score = 0
        for word in doc:
            doc_score += doc[word] * self.weights[label][word]

        return doc_score

    def predict(self, doc):
        """
        Return the highest-scoring label for the document under the current model.
        """
        max_score = 0
        high_label = ""
        for label in self.weights:
            doc_score = self.score(doc, label)
            if doc_score > max_score:
                max_score = doc_score
                high_label = label

        return high_label

    def test_eval(self, test_docs, test_labels):
        pred_labels = [self.predict(d) for d in test_docs]
        ev = Eval(test_labels, pred_labels)
        return ev.accuracy(), ev


    def highest_features(self, label):
        return self.weights[label].most_common(20)


    def lowest_features(self, label):
        return self.weights[label].most_common()[:-20-1:-1] 
        

    def learn(self, train_docs, train_labels):
        """
        Train on the provided data with the perceptron algorithm.
        Up to self.MAX_ITERATIONS of learning.
        At the end of training, self.weights should contain the final model
        parameters.
        """

        for i in range(0,self.MAX_ITERATIONS):
            
            updates = 0
            accuracy = 0

            for t in range(0,len(train_docs)):
                
                doc = train_docs[t]
                train_label = train_labels[t]

                max_score = -1
                predicted_label = ""
                
                for label in self.CLASSES:
                    total = 0
                    for word in doc:
                        total += doc[word] * self.weights[label][word]
                            
                    if total > max_score:
                        max_score = total
                        predicted_label = label

                if predicted_label != train_label: # predicted label is incorrect
                    updates += 1
                    
                    for word, count in doc.items():
                        # TODO: change this to be changed by the weight, not 1
                        self.weights[predicted_label][word] -= 1
                        self.weights[train_label][word] += 1
                else:
                    accuracy += 1

            # OUTPUT
            print("\nIteration: " + str(i))
            print("Updates: " + str(updates))
            print("Training Accuracy: " + str(accuracy/len(train_docs)) )

            if updates == 0:
                return

            # Test the dev accuracy
            dev_accuracy = 0
            for t in range(0,len(self.dev_docs)):

                doc = self.dev_docs[t]
                gold_label = self.dev_labels[t]

                max_score = -1
                predicted_label = ""
                
                for label in self.CLASSES:
                    total = 0
                    for word in doc:
                        total += doc[word] * self.weights[label][word]
                            
                    if total > max_score:
                        max_score = total
                        predicted_label = label

                if predicted_label == gold_label:
                    dev_accuracy += 1

            print("Dev Accuracy: " + str(dev_accuracy/len(dev_docs)))
        print('\n')


if __name__ == "__main__":
    start = time.time()
    global sentiment_classifier

    args = sys.argv[1:]
    niters = int(args[0])

    sentiment_classifier = NBClassifier()
    sentiment_classifier.train()

    train_docs, train_labels = load_featurized_docs('data/raw/train')
    print(len(train_docs), 'training docs with',
        sum(len(d) for d in train_docs)/len(train_docs), 'percepts on avg', file=sys.stderr)

    dev_docs,  dev_labels  = load_featurized_docs('data/raw/dev')
    print(len(dev_docs), 'dev docs with',
        sum(len(d) for d in dev_docs)/len(dev_docs), 'percepts on avg', file=sys.stderr)


    test_docs,  test_labels  = load_featurized_docs('data/raw/test')
    print(len(test_docs), 'test docs with',
        sum(len(d) for d in test_docs)/len(test_docs), 'percepts on avg', file=sys.stderr)

    ptron = Perceptron(train_docs, train_labels, MAX_ITERATIONS=niters, dev_docs=dev_docs, dev_labels=dev_labels)
    acc, ev = ptron.test_eval(test_docs, test_labels)
    print("Accuracy: " + str(acc))

    # Confusion Matrix
    print("\nConfusion Matrix: ")
    matrix = ev.confusion_matrix()
    print_matrix(matrix)

    for label in ptron.CLASSES:
        
        print("--------------------")
        print("\LABEL: " + label)

        # Highest and Lowest Features
        high_feats = ptron.highest_features(label)
        print("\nHighest weighted features:")
        for feat in high_feats:
            if type(feat[0]) is tuple:
                feature_string = ' '.join(feat[0])
            else:
                feature_string = feat[0]
            print(" - Feature: " + feature_string + "\t\t\tWeight: " + str(feat[1]))

        low_feats = ptron.lowest_features(label)
        print("\nLowest weighted features:")
        for feat in low_feats:
            if type(feat[0]) is tuple:
                feature_string = ' '.join(feat[0])
            else:
                feature_string = feat[0]
            print(" - Feature: " + feature_string + "\t\t\tWeight: " + str(feat[1]))

        print("Bias: " + str(ptron.weights[label]['bias']))

        print("\nPrecision: " + str(ev.precision(label)))

        print("\nRecall: " + str(ev.recall(label)))

        print("\nF1: " + str(ev.f1(label)))

        end = time.time()
        print(end - start)


