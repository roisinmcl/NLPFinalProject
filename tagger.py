from nltk import word_tokenize
from nltk.corpus import treebank
from nltk.tag import hmm


class HMMTagger():
    def __init__(self):
        self.trainer = hmm.HiddenMarkovModelTrainer()

    def train(self, TRAIN_DATA):
        self.tagger = self.trainer.train_supervised(self.read_tagged_corpus(TRAIN_DATA))

    def tag(self, sentence):
        return self.tagger.tag(sentence)

    def test(self, TEST_DATA):
        test_data = self.read_tagged_corpus(TEST_DATA)
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

    def read_tagged_sentence(self, f):
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

    def read_tagged_corpus(self, filename):
        sentences = []
        with open(filename, 'r', encoding='utf-8') as f:
            sentence = self.read_tagged_sentence(f)
            while sentence:
                sentences.append(sentence)
                sentence = self.read_tagged_sentence(f)
        return sentences

if __name__ == '__main__':
    TRAIN_DATA = 'data/en-ud-test.upos.tsv'
    TEST_DATA = 'data/en-ud-test.upos.tsv'

    tagger = HMMTagger()
    tagger.train(TRAIN_DATA)
    tagger.test(TEST_DATA)
