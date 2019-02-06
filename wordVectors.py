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

def saveVectors(trained_model, location, model='gensim', binary=True):
    if model == 'gensim':
        trained_model.save(location)
    elif model == 'w2v':
        trained_model.wv.save_word2vec_format(location, binary=binary)

def loadVectors(location, model='gensim', binary=True):
    if model == 'gensim':
        return Word2Vec.load(location)
    elif model == 'w2v':
        return KeyedVectors.load_word2vec_format(location, binary=binary)

def in_vocab(word, model):
	return word in model.wv.vocab

def get_vector(word, model):
	return model.wv[word]

def cosSim(wf1, wf2, model): # range 0, 1
	return model.wv.similarity(wf1, wf2)

def cosDist(wf1, wf2, model):
	return 1 - cosSim(wf1, wf2, model)

def dimensionality(model):
    return model.vector_size

def similar_by_vector(vec, model, topn=10):
    return model.wv.similar_by_vector(vec, topn)


# if __name__ == '__main__':
