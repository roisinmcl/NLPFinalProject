import os
import shutil

from tagger import PerceptronTagger
from util import write_tagged_sentence

if __name__ == '__main__':
    RAW_DIR = 'data/raw/'
    TAGGED_DIR = 'data/tagged/'

    tagger = PerceptronTagger(True)
    print('Using {} Tagger'.format(tagger.name))

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
                write_tagged_sentence(tagged_file, tagger.tag(sentence.split()))

            raw_file.close()
            tagged_file.close()
