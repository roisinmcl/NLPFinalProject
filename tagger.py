from nltk import word_tokenize
from nltk.corpus import treebank
from nltk.tag import hmm

TRAIN_DATA = 'data/en-ud-test.upos.tsv'
TEST_DATA = 'data/en-ud-test.upos.tsv'

def read_tagged_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != "\n"):
        line = line.strip()
        word, tag = line.split("\t", 2)
        sentence.append( (word, tag) )
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

trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(read_tagged_corpus(TRAIN_DATA))

test_data = read_tagged_corpus(TEST_DATA)
total = 0
correct = 0
for sentence in test_data:
    pred_tagging = tagger.tag(list(map(lambda x: x[0], sentence)))
    for (_, gold_tag), (_, pred_tag) in zip(sentence, pred_tagging):
        if gold_tag == pred_tag:
            correct += 1
    total += len(sentence)

print(correct / total)

# print(tagger.tag(word_tokenize('This is a test sentence.')))