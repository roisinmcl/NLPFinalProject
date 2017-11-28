import os
import shutil

from tagger import PerceptronTagger

if __name__ == '__main__':
    TRAIN_DATA = 'data/en-ud-train.upos.tsv'

    RAW_DIR = 'data/raw/'
    TAGGED_DIR = 'data/tagged/'

    print('Training Tagger')
    tagger = PerceptronTagger()
    tagger.train(TRAIN_DATA)

    if os.path.exists(TAGGED_DIR):
        shutil.rmtree(TAGGED_DIR)
    os.makedirs(TAGGED_DIR)

    subdirs = list(os.walk(RAW_DIR))[1:]
    for subdir in subdirs:
        subdir_name = subdir[0].rsplit('/', 1)[-1]
        os.makedirs(TAGGED_DIR + subdir_name)
        for filename in subdir[2]:
            print('Tagging', RAW_DIR + subdir_name + '/' + filename)
            raw_file = open(RAW_DIR + subdir_name + '/' + filename, 'r')
            tagged_file = open(TAGGED_DIR + subdir_name + '/' + filename, 'w')
            for sentence in raw_file:
                for token, tag in tagger.tag(sentence.split()):
                    tagged_file.write('{}\t{}\n'.format(token, tag))
                tagged_file.write('\n')
            raw_file.close()
            tagged_file.close()
