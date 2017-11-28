from nltk import word_tokenize, pos_tag
from nltk.tag import hmm, perceptron

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline


def read_tagged_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append((word, tag))
        line = f.readline()
    return sentence

def read_tagged_corpus(filename):
    sentences = []
    with open(filename, 'r', encoding='utf-8') as f:
        sentence = read_tagged_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_tagged_sentence(f)
    return sentences


class HMMTagger():
    def __init__(self):
        self.trainer = hmm.HiddenMarkovModelTrainer()

    def train(self, filename):
        self.tagger = self.trainer.train_supervised(read_tagged_corpus(filename))

    def tag(self, sentence):
        return self.tagger.tag(sentence)

    def test(self, filename):
        test_data = read_tagged_corpus(filename)
        correct_tags = 0
        total_tags = 0
        correct_sents = 0
        total_sents = len(test_data)
        for sentence in test_data:
            pred_tagging = self.tag(list(map(lambda x: x[0], sentence)))
            for (_, gold_tag), (_, pred_tag) in zip(sentence, pred_tagging):
                if gold_tag == pred_tag:
                    correct_tags += 1
            if sentence == pred_tagging:
                correct_sents += 1
            total_tags += len(sentence)
        print('Token Accuracy: {} / {} - {:.2f}%'.format(correct_tags, total_tags, correct_tags / total_tags * 100))
        print('Sentence Accuracy: {} / {} - {:.2f}%'.format(correct_sents, total_sents, correct_sents / total_sents * 100))


class DTCTagger():
    def __init__(self):
        self.clf = Pipeline([
            ('vectorizer', DictVectorizer(sparse=False)),
            ('classifier', DecisionTreeClassifier(criterion='entropy'))
        ])

    def train(self, filename):
        features, tags = self.transform(read_tagged_corpus(filename))
        self.clf.fit(features, tags)

    def tag(self, sentence):
        tags = self.clf.predict([self.get_features(sentence, i) for i in range(len(sentence))])
        return list(zip(sentence, tags))

    def test(self, filename):
        test_data = read_tagged_corpus(filename)
        correct_tags = 0
        total_tags = 0
        correct_sents = 0
        total_sents = len(test_data)
        for sentence in test_data:
            pred_tagging = self.tag(list(map(lambda x: x[0], sentence)))
            for (_, gold_tag), (_, pred_tag) in zip(sentence, pred_tagging):
                if gold_tag == pred_tag:
                    correct_tags += 1
            if sentence == pred_tagging:
                correct_sents += 1
            total_tags += len(sentence)
        print('Token Accuracy: {} / {} - {:.2f}%'.format(correct_tags, total_tags, correct_tags / total_tags * 100))
        print('Sentence Accuracy: {} / {} - {:.2f}%'.format(correct_sents, total_sents, correct_sents / total_sents * 100))

    def get_features(self, sentence, index):
        return {
            'word': sentence[index],
            'capitalized': sentence[index][0].isupper(),
            'uppercase': sentence[index].isupper(),
            'lowercase': sentence[index].islower(),
            'first': index == 0,
            'last': index == len(sentence) - 1,
            'prefix_1': sentence[index][0],
            'prefix_2': sentence[index][:2],
            'prefix_3': sentence[index][:3],
            'prefix_4': sentence[index][:4],
            'suffix_1': sentence[index][-1],
            'suffix_2': sentence[index][-2:],
            'suffix_3': sentence[index][-3:],
            'suffix_4': sentence[index][-4:],
            'previous': sentence[index - 1] if index != 0 else '',
            'next': sentence[index + 1] if index != len(sentence) - 1 else '',
            'has_hyphen': '-' in sentence[index],
            'numeric': sentence[index].isdigit(),
            'camal_case': sentence[index][1:] != sentence[index][1:].lower()
        }

    def transform(self, sentences):
        features, tags = [], []
        for sentence in sentences:
            for i in range(len(sentence)):
                features.append(self.get_features([word for (word, _) in sentence], i))
                tags.append(sentence[i][1])
        return features, tags


class PerceptronTagger():
    def __init__(self):
        self.tagger = perceptron.PerceptronTagger(load=False)

    def train(self, filename):
        self.tagger.train(read_tagged_corpus(filename))

    def tag(self, sentence):
        return self.tagger.tag(sentence)

    def test(self, filename):
        test_data = read_tagged_corpus(filename)
        correct_tags = 0
        total_tags = 0
        correct_sents = 0
        total_sents = len(test_data)
        for sentence in test_data:
            pred_tagging = self.tag(list(map(lambda x: x[0], sentence)))
            for (_, gold_tag), (_, pred_tag) in zip(sentence, pred_tagging):
                if gold_tag == pred_tag:
                    correct_tags += 1
            if sentence == pred_tagging:
                correct_sents += 1
            total_tags += len(sentence)
        print('Token Accuracy: {} / {} - {:.2f}%'.format(correct_tags, total_tags, correct_tags / total_tags * 100))
        print('Sentence Accuracy: {} / {} - {:.2f}%'.format(correct_sents, total_sents, correct_sents / total_sents * 100))


if __name__ == '__main__':
    TRAIN_DATA = 'data/en-ud-train.upos.tsv'
    TEST_DATA = 'data/en-ud-test.upos.tsv'
    DEV_DATA = 'data/en-ud-dev.upos.tsv'

    tagger = PerceptronTagger()
    print('Training Model')
    tagger.train(TRAIN_DATA)
    print('Training Complete')
    print('Testing Model')
    tagger.test(TEST_DATA)
    tagger.test(DEV_DATA)
