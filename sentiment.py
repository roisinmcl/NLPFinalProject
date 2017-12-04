import requests
import json

def get_sentiment(word):
	text = {"text": word}
	response = requests.post("http://text-processing.com/api/sentiment/", data=text)
	data = response.json()
	print(data)
	if data['probability']['pos'] > 0.6:
		return 'pos'
	elif data['probability']['neg'] > 0.6:
		return 'neg'
	else:
		return 'neutral'