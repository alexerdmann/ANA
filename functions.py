from sys import argv, stdout, exit

import numpy as np
import Levenshtein as lv
import statistics
import os
import time
import random

def normalize(model):
	denom = 0
	for x in model:
		denom += model[x]
	for x in model:
		model[x] /= denom

	return model

def normalize_binary(model):
	denom = 0
	for x in model:
		model[x] = model[x][0] / model[x][1]

	return model

def smoothUNK_normalize(model):
	denom = 0
	types = len(model)
	for x in model:
		denom += model[x]
	for x in model:
		# model[x] += 1
		# model[x] /= (denom + types)
		model[x] /= denom
	model['<UNK>'] = min(list(model.values()))
	model['<TYPES>'] = types
	model['<TOKENS>'] = denom

	return model

def model_lookup(model, keys):

	while len(keys) > 0:
		key = keys.pop(0)
		if key not in model:

			if len(keys) == 0:
				return model['<UNK>']
			else:
				return None

		model = model[key]

	value = model

	return value

def get_features_from_array(ANAarray, ANAskeleton, lemma, wf):
	allFts = []
	for i in range(len(ANAarray[lemma][wf])):
		fts = []
		for dimInd in range(len(ANAarray[lemma][wf][i])):
			fts.append(ANAskeleton[dimInd][ANAarray[lemma][wf][i][dimInd]])
		allFts.append(fts)

	return allFts

### check that base and exponent characters are weighted ideally
	# debug_check_baseExponent_probs(ana)
def debug_check_baseExponent_probs(ana):
	l = []
	for ch in ana.exponenceWeights:
		l.append([ana.exponenceWeights[ch], ch])
	l.sort(reverse=True)
	print('MOST LIKELY EXPONENT LETTERS UNIVERSALLY:')
	for i in range(10):
		print('{}\t{}'.format(l[i][1], str(round(l[i][0], 3))))
	print()

	p1 = 'ابتعد'
	print('FOR PARADIGM {}...'.format(p1))

	l = []
	for ch in ana.baseWeights[p1]:
		l.append([ana.baseWeights[p1][ch], ch])
	l.sort(reverse=True)
	print('MOST LIKELY BASE LETTERS:')
	for i in range(10):
		print('{}\t{}'.format(l[i][1], str(round(l[i][0], 3))))
	print()

	l = []
	for ch in ana.condExpWeight[p1]:
		l.append([ana.condExpWeight[p1][ch], ch])
	l.sort(reverse=True)
	print('MOST LIKELY EXPONENT LETTERS:')
	for i in range(10):
		print('{}\t{}'.format(l[i][1], str(round(l[i][0], 3))))
	print()

### evaluate accuracy in assigning lower probability to non-cell mates
	# [control/our]Accuracy = debug_check_paradigmPair_cellMate_accuracies(ana, [True/False])
def debug_check_paradigmPair_cellMate_accuracies(ana, control):
	# p1 = 'ابتعد' # ara
	# p2 = 'آثر'
	# p1 = 'ächzen' # deu
	p1 = 'abändern'
	p2 = 'abstoßen'
	totalAccuracy = []

	# for every word in p1
	for wf1 in ana.data[p1]:
		tag1 = ana.data[p1][wf1][0]
		allScores = []
		# for every word in p2
		for wf2 in ana.data[p2]:
			tag2 = ana.data[p2][wf2][0]
			# get the probability that they are cell mates
			if control:
				PcellMates = ana.get_likelihood_cell_mate_control(p1, wf1, p2, wf2)
			else:
				PcellMates = ana.get_likelihood_cell_mate(p1, wf1, p2, wf2)
			allScores.append(PcellMates)
			# store the probability recorded by the actual cell mates
			if tag1 == tag2:
				matchProb = PcellMates
		allScores.sort(reverse=True)

		# see how many words scored higher than the true cell mate
		wrong = 0
		while allScores[0] > matchProb:
			allScores = allScores[1:]
			wrong += 1
		correct = len(allScores)
		for score in allScores:
			if score == matchProb:
				correct -= 0.5
				wrong += 0.5
			else:
				break

		# print and store accuracies for each word form in p1
		accuracy = correct / (correct + wrong)
		print('Accuracy {}:\t{}'.format(''.join(wf1), str(accuracy)))
		totalAccuracy.append(accuracy)

	# calculate the aggregate accuracy over the entire paradigm
	overallAccuracy = sum(totalAccuracy)/len(totalAccuracy)
	print('OVERALL ACCURACY: {}'.format(str(round(overallAccuracy, 3))))

	return overallAccuracy

def debug_ensure_correct_feature_mapping(ana):
	for lemma in ana.GOLD_array:
		for wf in ana.GOLD_array[lemma]:
			allFts = get_features_from_array(ana.GOLD_array, ana.GOLD_skeleton, lemma, wf)
			for fts in allFts:
				fts = ';'.join(fts)
				fts2 = fts.replace(';',' ').replace('NONE','')
				fts2 = ';'.join(fts2.split())
				# print('{}\t{}\n\t{}'.format(wf, lemma, fts))
				# print('\t{}'.format(fts2))
				# for i in range(len(ana.GOLD_data[lemma][wf])):
					# print('\t{}'.format(ana.GOLD_data[lemma][wf][i]))
				assert fts2.split(';') in ana.GOLD_data[lemma][wf]

def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids\n\tvalid medoids: {}\n\tinvalid medoids: {}\n\trequested clusters: {}'.format(len(valid_medoid_inds), len(invalid_medoid_inds), k))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C
	# C is a dictionary from cluster labels to an array of member ind's
	# M is a list of kemoid-center ind's with indeces corresponding to cluster labels


