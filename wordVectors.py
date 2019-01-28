from sys import argv, exit, stdin, stdout, stderr
import os
import time
import optparse
import numpy as np
import gensim

### for word2vec
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import Text8Corpus

### for gensim's fasttext implementation
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim


def trainVectors(corpus, implementation, dim=300, min_n=3, max_n=6, min_count=1, model='skipgram', epochs=5, threads=12, window=5, lr=0.05, t=1e-4, neg=5):

    if implementation == 'fasttext':

        ### PARSE TRAIN DATA
        train_data = LineSentence(corpus)
        ### INITIALIZE MODEL
        model_gensim = FT_gensim(size=dim, min_n=min_n, max_n=max_n, min_count=min_count, iter=epochs, window=window)
        # BUILD VOCABULARY
        model_gensim.build_vocab(train_data)
        ### TRAIN THE MODEL
        model_gensim.train(train_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter, model=model, threads=threads, lr=lr, t=t, neg=neg)

    elif implementation == 'w2v':

    	### PARSE TRAIN DATA
        train_data = LineSentence(corpus)
        ### TRAIN THE MODEL
        model_gensim = Word2Vec(size=dim, min_count=min_count, iter=epochs, window=window, workers=threads)
        # BUILD VOCABULARY
        model_gensim.build_vocab(train_data)
        ### TRAIN THE MODEL
        model_gensim.train(train_data, total_examples=model_gensim.corpus_count, epochs=model_gensim.iter)

    return model_gensim

def saveVectors(trained_model, location):
    trained_model.save(location)

def loadVectors(location):
    return Word2Vec.load(location)

def in_vocab(word, model):
	return word in model.wv.vocab

def get_vector(word, model):
	return model.wv[word]

def cosSim(wf1, wf2, model): # range 0, 1
	return model.wv.similarity(wf1, wf2)

def cosDist(wf1, wf2, model):
	return 1 - cosSim(wf1, wf2, model)


# if __name__ == '__main__':
#     ### NOT FINISHED YET

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-corpus', type=str, help='Raw corpus to train embeddings on', required=False)
#     parser.add_argument('-model', type=str, help='Which word vector model to use', choices=['w2v','fasttext'], required=True)
#     parser.add_argument('-output', type=str, help='Where to save trained vectors', required=True)
#     parser.add_argument('-action', type=str, help='What do you want to do?', choices=['train', 'query'], required=True)
#     parser.add_argument('-dim', type=int, help='Dimensions of word vectors', required=False, default=300)
#     args = parser.parse_args()

#     ### TEST SET UP WITH LEE CORPUS
#     args.corpus = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
#     args.corpus = args.corpus + 'lee_background.cor'
#     ###

#     if args.model == 'fasttext':
#         model_gensim = train_FT(args)
#     elif args.model == 'w2v':
#         model_gensim = train_w2v(args)

#     print(model_gensim)