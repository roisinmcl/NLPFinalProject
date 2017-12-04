import requests
import json

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