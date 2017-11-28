from nltk import word_tokenize, pos_tag
from nltk.tag import hmm, perceptron


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

def test(tagger, filename):
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


class HMMTagger():
    def __init__(self):
        self.trainer = hmm.HiddenMarkovModelTrainer()

    def train(self, filename):
        self.tagger = self.trainer.train_supervised(read_tagged_corpus(filename))

    def tag(self, sentence):
        return self.tagger.tag(sentence)

class PerceptronTagger():
    def __init__(self):
        self.tagger = perceptron.PerceptronTagger(load=False)

    def train(self, filename):
        self.tagger.train(read_tagged_corpus(filename))

    def tag(self, sentence):
        return self.tagger.tag(sentence)


if __name__ == '__main__':
    TRAIN_DATA = 'data/en-ud-train.upos.tsv'
    TEST_DATA = 'data/en-ud-test.upos.tsv'
    DEV_DATA = 'data/en-ud-dev.upos.tsv'

    tagger = PerceptronTagger()
    tagger.train(TRAIN_DATA)
    test(tagger, TEST_DATA)
    test(tagger, DEV_DATA)
