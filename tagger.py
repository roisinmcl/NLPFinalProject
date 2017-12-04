from nltk import word_tokenize, pos_tag
from nltk.tag import hmm, perceptron
try: # spacy can be tricky to install
    import spacy
except ImportError:
    pass

from util import read_tagged_corpus


class Tagger():
    def __init__(self, name):
        self.name = name

    def test(self, filename):
        test_data = read_tagged_corpus(filename)
        correct_tags = 0
        total_tags = 0
        correct_sents = 0
        total_sents = len(test_data)
        for sentence in test_data:
            pred_tagging = tagger.tag(list(map(lambda x: x[0], sentence)))
            for (_, gold_tag), (_, pred_tag) in zip(sentence, pred_tagging):
                if gold_tag == pred_tag:
                    correct_tags += 1
            if sentence == pred_tagging:
                correct_sents += 1
            total_tags += len(sentence)
        print('Token Accuracy: {} / {} - {:.2f}%'.format(correct_tags, total_tags, correct_tags / total_tags * 100))
        print('Sentence Accuracy: {} / {} - {:.2f}%'.format(correct_sents, total_sents, correct_sents / total_sents * 100))


class HMMTagger(Tagger):
    def __init__(self):
        super().__init__('HMM')
        self.trainer = hmm.HiddenMarkovModelTrainer()

    def train(self, filename):
        self.tagger = self.trainer.train_supervised(read_tagged_corpus(filename))

    def tag(self, sentence):
        return self.tagger.tag(sentence)

    def test(self, filename):
        return super().test(filename)


class PerceptronTagger(Tagger):
    def __init__(self, load):
        super().__init__('Perceptron')
        self.tagger = perceptron.PerceptronTagger(load=load)

    def train(self, filename):
        self.tagger.train(read_tagged_corpus(filename))

    def tag(self, sentence):
        return self.tagger.tag(sentence)

    def test(self, filename):
        return super().test(filename)


class SpacyTagger(Tagger):
    def __init__(self):
        super().__init__('Spacy')
        self.nlp = spacy.load('en')

    def train(self, filename):
        pass

    def tag(self, sentence):
        return [(token.text, token.pos_) for token in self.nlp(' '.join(sentence))]

    def test(self, filename):
        return super().test(filename)
