def read_tagged_sentence(f):
    line = f.readline()
    if not line:
        return None
    sentence = []
    while line and (line != '\n'):
        line = line.strip()
        token, tag = line.split('\t', 2)
        sentence.append((token, tag))
        line = f.readline()
    return sentence

def read_tagged_corpus(filename):
    sentences = []
    with open(filename, 'r') as f:
        sentence = read_tagged_sentence(f)
        while sentence:
            sentences.append(sentence)
            sentence = read_tagged_sentence(f)
    return sentences

def write_tagged_sentence(f, sentence):
    for token, tag in sentence:
        f.write('{}\t{}\n'.format(token, tag))
    f.write('\n')

def write_tagged_corpus(filename, sentences):
    with open(filename, 'w') as f:
        for sentence in sentences:
            write_tagged_sentence(f, sentence)
