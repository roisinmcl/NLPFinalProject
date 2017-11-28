import os
import shutil

from tagger import PerceptronTagger

if __name__ == '__main__':
    TRAIN_DATA = 'data/en-ud-train.upos.tsv'

    RAW_DIR = 'data/raw/'
    TAGGED_DIR = 'data/tagged/'

    # tagger = PerceptronTagger()
    # tagger.train(TRAIN_DATA)

    if os.path.exists(TAGGED_DIR):
        shutil.rmtree(TAGGED_DIR)
    os.makedirs(TAGGED_DIR)

    subdirs = list(os.walk(RAW_DIR))[1:]
    for subdir in subdirs:
        subdir_name = subdir[0].rsplit('/', 1)[-1]
        os.makedirs(TAGGED_DIR + subdir_name)
        for filename in subdir[2]:
            with open(TAGGED_DIR + subdir_name + '/' + filename, 'w') as f:
                f.write(filename)
