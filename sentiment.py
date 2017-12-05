import requests
import json

from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews

class NBClassifier():
	def __init__(self):
		self.classifier = None

		neg_features = [({word: True}, 'neg') for word in movie_reviews.fileids('neg')]
		pos_features = [({word: True}, 'pos') for word in movie_reviews.fileids('pos')]
		neg_cutoff = int(len(neg_features) * 3 / 4)
		pos_cutoff = int(len(pos_features) * 3 / 4)

		self.train_features = neg_features[:neg_cutoff] + pos_features[:pos_cutoff]
		self.test_features = neg_features[neg_cutoff:] + pos_features[pos_cutoff:]

	def train(self):
		self.classifier = NaiveBayesClassifier.train(self.train_features)

	def get_sentiment(self, word):
		prob_dict = self.classifier.prob_classify({word: True})
		if prob_dict.prob('pos') > 0.6:
			return 5
		elif prob_dict.prob('neg') > 0.6:
			return -5
		else:
			return 0

	def test(self):
		print('Accuracy: {}'.format(accuracy(self.classifier, self.test_features)))

def get_sentiment(word):
	text = {"text": word}
	response = requests.post("http://text-processing.com/api/sentiment/", data=text)
	try:
		data = response.json()
		#print(data)
		if data['probability']['pos'] > 0.6:
			return 5
		elif data['probability']['neg'] > 0.6:
			return -5
		else:
			return 0
	except json.decoder.JSONDecodeError:
		print("JSONDecodeError occured")
		return 0
