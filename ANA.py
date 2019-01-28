from sys import argv, stdout, exit

import argparse
import numpy as np
import Levenshtein as lv
import statistics
import os
import time
import random
from functions import *
from wordVectors import *
import pickle as pkl
import operator as op
from functools import reduce
import itertools
from sklearn.metrics import silhouette_samples


ANA_DIR = os.path.dirname(os.path.realpath(argv[0]))
UNIMORPH_UG = os.path.join(ANA_DIR, 'Unimorph/unimorph_feature_dimensions.tsv')


class ANA:

    def __init__(self, fn):
        

        ### get UG dimensions of features
        self.UG_dim_to_feats = {}
        self.UG_feats_to_dim = {}

        for line in open(UNIMORPH_UG):
            dim, ft = line.split()
            ft = ft.upper()
            if dim not in self.UG_dim_to_feats:
                self.UG_dim_to_feats[dim] = {}
            self.UG_dim_to_feats[dim][ft] = True
            if ft not in self.UG_feats_to_dim:
                self.UG_feats_to_dim[ft] = dim


        ### get gold data, feat-dim maps, no-fts data, dimension order
        self.GOLD_data = {}
        self.dimOrder = []
        self.data = {}
        self.GOLD_dim_to_feats = {}
        self.GOLD_feats_to_dim = {}

        for line in open(fn):
            line = line.strip('\n')
            if line:
                line = line.split('\t')
                lemma, wf, fts = map(str, line)

                ## record gold and featureless data
                if lemma not in self.GOLD_data:
                    self.GOLD_data[lemma] = {}
                    self.data[lemma] = {}
                if wf not in self.GOLD_data[lemma]:
                    self.GOLD_data[lemma][wf] = []
                    self.data[lemma][wf] = True

                ## handle feature notation inconsistencies
                fts = fts.replace('NDEF','INDF')
                fts = fts.split(';')
                # check for multiple feats per dimension
                type_dim2fts = {}
                to_remove = {}
                for ft in fts:
                    dim = self.UG_feats_to_dim[ft]
                    if dim not in type_dim2fts:
                        type_dim2fts[dim] = []
                    type_dim2fts[dim].append(ft)
                    if len(type_dim2fts[dim]) > 1:
                        assert len(type_dim2fts[dim]) == 2
                        prefix = type_dim2fts[dim][0].split('.')[0]
                        assert prefix == type_dim2fts[dim][1].split('.')[0]
                        to_remove[prefix] = True
                if len(to_remove) > 0:
                    new_fts = []
                    for ft in fts:
                        if ft not in to_remove:
                            new_fts.append(ft)
                    fts = new_fts
                self.GOLD_data[lemma][wf].append(fts)

                ## induce preferred order of feature dimensions
                lowestIndex = 0
                for ft in fts:
                    dim = self.UG_feats_to_dim[ft]
                    if dim not in self.GOLD_dim_to_feats:
                        self.GOLD_dim_to_feats[dim] = {}
                    if ft not in self.GOLD_dim_to_feats[dim]:
                        self.GOLD_dim_to_feats[dim][ft] = True
                        self.GOLD_feats_to_dim[ft] = dim
                    if dim in self.dimOrder:
                        dimIndex = self.dimOrder.index(dim)
                        # we need to rearrange
                        if dimIndex < lowestIndex:
                            self.dimOrder.pop(dimIndex)
                            self.dimOrder.insert(lowestIndex, dim)
                        # we're kosher
                        else:
                            lowestIndex = dimIndex
                    # we need to add the dimension in
                    else:
                        self.dimOrder.insert(lowestIndex, dim)
                    lowestIndex += 1
        assert len(self.dimOrder) == len(self.GOLD_dim_to_feats)


        ### get UG skeleton and UG feat to skeleton map
        self.UG_feats_to_skel_coordinates = {}
        self.UG_skeleton = []

        for dimInd in range(len(self.dimOrder)):
            dim = self.dimOrder[dimInd]
            featInd = 0
            self.UG_skeleton.append(['NONE'])
            for ft in self.UG_dim_to_feats[dim]:
                featInd += 1
                self.UG_skeleton[-1].append(ft)
                self.UG_feats_to_skel_coordinates[ft] = [dimInd, featInd]


        ### get gold skeleton and feature map with consistent ordering
        ## limit gold skeleton to only UG features relevant in this lg
        self.GOLD_skeleton = []
        self.GOLD_feats_to_skel_coordinates = {}

        for dimInd in range(len(self.dimOrder)):
            dim = self.dimOrder[dimInd]
            self.GOLD_skeleton.append(['NONE'])
            featInd = 0
            for ft in self.GOLD_dim_to_feats[dim]:
                self.GOLD_skeleton[-1].append(ft)
                featInd += 1
                self.GOLD_feats_to_skel_coordinates[ft] = [dimInd, featInd]


        ### get gold array of paradigms
        self.GOLD_array = {}

        for lemma in self.GOLD_data:
            self.GOLD_array[lemma] = {}
            for wf in self.GOLD_data[lemma]:
                self.GOLD_array[lemma][wf] = []
                for fts in self.GOLD_data[lemma][wf]:
                    self.GOLD_array[lemma][wf].append([0] * len(self.GOLD_skeleton))
                    for ft in fts:
                        coord = self.GOLD_feats_to_skel_coordinates[ft]
                        assert ft == self.GOLD_skeleton[coord[0]][coord[1]]
                        self.GOLD_array[lemma][wf][-1][coord[0]] = coord[1]


        ## integerize everything
        self.wf2lem = {}
        self.lem2wf = {}        
        self.ind2wf = {}
        self.wf2ind = {}
        self.lem2ind = {}
        self.ind2lem = {}

        ind = -1
        deterministicLemOrder = list(self.data)
        deterministicLemOrder.sort()
        for lem in deterministicLemOrder:
            deterministicWForder = list(self.data[lem])
            deterministicWForder.sort()
            for wf in deterministicWForder:
                ind += 1
                ## manage wf's
                if wf not in self.wf2lem:
                    self.wf2lem[wf] = {}
                self.wf2lem[wf][lem] = True
                if wf not in self.wf2ind:
                    self.wf2ind[wf] = {}
                self.wf2ind[wf][ind] = True
                ## manage lemmas's
                if lem not in self.lem2wf:
                    self.lem2wf[lem] = {}
                self.lem2wf[lem][wf] = True
                if lem not in self.lem2ind:
                    self.lem2ind[lem] = {}
                self.lem2ind[lem][ind] = True
                ## manage ind's
                self.ind2wf[ind] = wf
                self.ind2lem[ind] = lem


    def learn_exponence_weights(self):  ### TO ADD: LINEAR INTERPOLATION WITH NGRAM EDIT BLOCKS

        try:
            self.baseWeights = pkl.load(open('{}.baseWeights.pkl'.format(UNIMORPH_LG), 'rb'))
            print('Reading cached paradigm conditional exponence weights')
            self.exponenceWeights = pkl.load(open('{}.exponenceWeights.pkl'.format(UNIMORPH_LG), 'rb'))
            self.condExpWeight = pkl.load(open('{}.condExpWeight.pkl'.format(UNIMORPH_LG), 'rb'))

        except FileNotFoundError:
            print('Learning paradigm conditional exponence weights')
            self.baseWeights = {}
            self.exponenceWeights = {}
            self.condExpWeight = {}

            ### get edit distance between all possible paradigm mate pairs
            counter = 0
            for lemma in self.data:
                self.baseWeights[lemma] = {}

                eligibleWF2 = dict(self.data[lemma])
                for wf1 in self.data[lemma]:
                    del eligibleWF2[wf1]
                    for wf2 in eligibleWF2:

                        ### keep track of how often each character occurred
                        for wf in wf1, wf2:
                            # both unconditioned and conditioned on lemma
                                # the unconditioned will be relevant for exponents
                                    # as these should be similar across paradigms
                                # the lemma conditioned will be relevant for bases
                                    # as these should be similar within paradigms
                            for condition in self.baseWeights[lemma], self.exponenceWeights:
                                for ch in wf:
                                    if ch not in condition:
                                        condition[ch] = [0, 0] # [#(edited), #(occured)]
                                    condition[ch][1] += 1

                        ### and how likely it is to be edited
                            ## conditional on paradigm and absolutely
                        for op in lv.editops(''.join(wf1), ''.join(wf2)):
                            # both word forms' unaligned characters
                            if op[0] == 'replace':
                                for ch in wf1[op[1]], wf2[op[2]]:
                                    for condition in self.baseWeights[lemma], self.exponenceWeights:
                                        condition[ch][0] += 1

                            # word form 1's unaligned characters
                            elif op[0] == 'delete':
                                ch = wf1[op[1]]
                                for condition in self.baseWeights[lemma], self.exponenceWeights:
                                    condition[ch][0] += 1

                            # word form 2's unaligned characters
                            elif op[0] == 'insert':
                                ch = wf2[op[2]]
                                for condition in self.baseWeights[lemma], self.exponenceWeights:
                                    condition[ch][0] += 1

                        counter += 1

            ### normalize the base weights over all lemmas
                ## inverse the probabilities to get conditional prob of bases
                    # (otherwise, it's just a conditional exponent probability)
            for lemma in self.baseWeights:

                ## exception handling for singleton paradigms
                if len(self.baseWeights[lemma]) == 0:
                    assert len(self.data[lemma]) == 1
                    for wf in self.data[lemma]:
                        for ch in wf:
                            if ch not in self.baseWeights[lemma]:
                                self.baseWeights[lemma][ch] = [0, 0]
                            self.baseWeights[lemma][ch][1] += 1
                            if ch not in self.exponenceWeights: # ch only occurs in singleton paradigm
                                self.exponenceWeights[ch] = [0, 1] # definitely not part of an exponent

                ## calculate base and exponent likelihoods as function of edit frequencies
                self.baseWeights[lemma] = normalize_binary(self.baseWeights[lemma])
                for x in self.baseWeights[lemma]:
                    self.baseWeights[lemma][x] = 1-self.baseWeights[lemma][x]
            self.exponenceWeights = normalize_binary(self.exponenceWeights)

            ### learn the likelihood that any character is involved in exponent
                ## conditional on paradigm membership
            for lemma in self.baseWeights:
                self.condExpWeight[lemma] = {}
                for ch in self.baseWeights[lemma]:
                    self.condExpWeight[lemma][ch] = self.exponenceWeights[ch] / (self.exponenceWeights[ch] + self.baseWeights[lemma][ch])

            ### cache learned weights
            pkl.dump( self.baseWeights, open('{}.baseWeights.pkl'.format(UNIMORPH_LG), 'wb' ) )
            pkl.dump( self.exponenceWeights, open('{}.exponenceWeights.pkl'.format(UNIMORPH_LG), 'wb' ) )
            pkl.dump( self.condExpWeight, open('{}.condExpWeight.pkl'.format(UNIMORPH_LG), 'wb' ) )


    def get_likelihood_cell_mate_bad(self, p1, wf1, p2, wf2):

        # for both words, for every char in word
            # add condExpWeight to denominator
        # for every matched char, for both words,
            # add condExpWeight to numerator

        ### comment this out to promote overabundance in weightedLev metric
        if p1 == p2:
            return 0.0

        # get all matched blocks
        blocks = lv.matching_blocks(lv.editops(''.join(wf1), ''.join(wf2)), len(wf1), len(wf2))

        # mark which characters matched
        match1 = [False] * len(wf1)
        match2 = [False] * len(wf2)
        for match in blocks:
            for i in range(match[0], match[0]+match[2]):
                match1[i] = True
            for i in range(match[1], match[1]+match[2]):
                match2[i] = True

        # sum numerator and denominator
        numerator = 0.0
        denominator = 0.0000001
        # over word 1
        for i in range(len(match1)):
            if match1[i]:
                ch = wf1[i]
                numerator += self.exponenceWeights[ch]
                denominator += self.baseWeights[p1][ch] + self.exponenceWeights[ch]
        # over word 2
        for i in range(len(match2)):
            if match2[i]:
                ch = wf2[i]
                numerator += self.exponenceWeights[ch]
                denominator += self.baseWeights[p2][ch] + self.exponenceWeights[ch]

        similarity = numerator/denominator
        
        return similarity


    def get_likelihood_cell_mate_2(self, p1, wf1, p2, wf2):

        # for both words, for every char in word
            # add condExpWeight to denominator
        # for every matched char, for both words,
            # add condExpWeight to numerator

        ### comment this out to promote overabundance in weightedLev metric
        if p1 == p2:
            return 0.0

        # get all matched blocks
        blocks = lv.matching_blocks(lv.editops(''.join(wf1), ''.join(wf2)), len(wf1), len(wf2))

        # mark which characters matched
        match1 = [False] * len(wf1)
        match2 = [False] * len(wf2)
        for match in blocks:
            for i in range(match[0], match[0]+match[2]):
                match1[i] = True
            for i in range(match[1], match[1]+match[2]):
                match2[i] = True

        # sum numerator and denominator
        numerator = 0.0
        denominator = 0.0000001
        # over word 1
        for i in range(len(match1)):
            ch = wf1[i]
            denominator += self.condExpWeight[p1][ch]
            if match1[i]:
                numerator += self.condExpWeight[p1][ch]
        # over word 2
        for i in range(len(match2)):
            ch = wf2[i]
            denominator += self.condExpWeight[p2][ch]
            if match2[i]:
                numerator += self.condExpWeight[p2][ch]

        similarity = numerator/denominator
        
        return similarity


    def get_likelihood_cell_mate(self, p1, wf1, p2, wf2):

        # for both words, for every char in word
            # add condExpWeight to denominator
        # for every matched char, for both words,
            # add condExpWeight to numerator

        ### comment this out to promote overabundance in weightedLev metric
        if p1 == p2:
            return 0.0

        # get all matched blocks
        blocks = lv.matching_blocks(lv.editops(''.join(wf1), ''.join(wf2)), len(wf1), len(wf2))

        # mark which characters matched
        match1 = [False] * len(wf1)
        match2 = [False] * len(wf2)
        for match in blocks:
            for i in range(match[0], match[0]+match[2]):
                match1[i] = True
            for i in range(match[1], match[1]+match[2]):
                match2[i] = True

        # sum numerator and denominator
        numerator = 0.0
        denominator = 0.0000001
        # over word 1
        for i in range(len(match1)):
            ch = wf1[i]
            denominator += self.condExpWeight[p1][ch]
            if match1[i]:
                numerator += self.condExpWeight[p1][ch]
        # over word 2
        for i in range(len(match2)):
            ch = wf2[i]
            denominator += self.condExpWeight[p2][ch]
            if match2[i]:
                numerator += self.condExpWeight[p2][ch]

        similarity = numerator/denominator
        
        return similarity


    def get_likelihood_cell_mate_control(self, p1, wf1, p2, wf2):

        numerator = 0
        denominator = len(wf1) + len(wf2)
        blocks = lv.matching_blocks(lv.editops(''.join(wf1), ''.join(wf2)), len(wf1), len(wf2))
        for match in blocks:
            for i in range(match[0], match[0]+match[2]):
                numerator += 1

        return numerator/denominator


    def get_possible_coordinates(self):

        self.all_possible_coordinates = [[]]

        dimInd = -1
        while dimInd < len(self.GOLD_skeleton)-1:
            dimInd += 1
            new_total_possible_coordinates = []
            for coord_so_far in self.all_possible_coordinates:
                for featInd in range(len(self.GOLD_skeleton[dimInd])):
                    new_possible_coordinate = coord_so_far[:]
                    new_possible_coordinate.append(featInd)
                    new_total_possible_coordinates.append(new_possible_coordinate)
            self.all_possible_coordinates = new_total_possible_coordinates[:]


    def get_attested_coordinates(self):

        ### get all attested coordinates and word classes
        self.total_coordinates = {}
        lemma_to_coords = {}
        self.total_wordClasses = {}
        self.strCoord_lstCoord = {}
        self.GOLD_cell_wf = {}

        for lemma in self.GOLD_array:
            if lemma not in lemma_to_coords:
                lemma_to_coords[lemma] = {}
            for wf in self.GOLD_array[lemma]:
                for realization in self.GOLD_array[lemma][wf]:
                    coord = ','.join(str(x) for x in realization)

                    if coord not in self.GOLD_cell_wf:
                        self.GOLD_cell_wf[coord] = {}
                    self.GOLD_cell_wf[coord][wf] = True

                    self.strCoord_lstCoord[coord] = []
                    for f in range(len(realization)):
                        self.strCoord_lstCoord[coord].append('{},{}'.format(f, realization[f]))

                    self.total_coordinates[coord] = True
                    lemma_to_coords[lemma][coord] = True
            all_coords = list(lemma_to_coords[lemma].keys())
            all_coords.sort()
            all_coords = '\n'.join(all_coords)
            self.total_wordClasses[all_coords] = True


        ### map all coordinates to possible wordClasses and possible paradigm mates
        self.coords_to_wordClasses = {}
        self.coord_syncretisms = {}

        for wc in self.total_wordClasses:
            wcList = wc.split('\n')
            for coordID in range(len(wcList)):
                coord = wcList[coordID]
                if coord not in self.coords_to_wordClasses:
                    self.coords_to_wordClasses[coord] = {}
                    self.coord_syncretisms[coord] = {}
                self.coords_to_wordClasses[coord][wc] = True
                for coordID2 in range(len(wcList)):
                    if coordID2 != coordID:
                        coord2 = wcList[coordID2]
                        self.coord_syncretisms[coord][coord2] = True


    def assignCellMates_random(self):

        print('Randomly assigning forms to cells')

        self.array = {}
        
        for lemma in self.data:
            self.assignParadigm_random(lemma)

        cellMates, coordinates_to_forms, forms_to_coordinates = get_cellMates(ana.array)

        return cellMates


    def assignParadigm_random(self, lemma):

        self.array[lemma] = {}
        random_coords = random.sample(list(self.total_coordinates), len(self.data[lemma]))
        for wf in self.data[lemma]:
            coord = random_coords.pop()
            listCord = [int(x) for x in coord.split(',')]
            self.array[lemma][wf] = [listCord]


    def get_distance_matrix(self, distance_function, embeddings=None, D1= None, D2=None, alpha=None):

        n = len(self.ind2wf)
        if distance_function == 'interpolated':
            filename = '{}.{}_{}-{}-{}_distMatrix.dat'.format(UNIMORPH_LG, distance_function, D1, D2, str(alpha))
        else:
            filename = '{}.{}_distMatrix.dat'.format(UNIMORPH_LG, distance_function)

        try:
            print('Checking for cached {}x{} {} distance matrix'.format(str(n), str(n), distance_function))
            # matrix1 = np.memmap(filename, dtype='float64', mode='r', shape=(n, n))
            matrix = np.load(filename)

        except FileNotFoundError:

            print('Building {}x{} {} distance matrix'.format(str(n), str(n), distance_function))
            matrix = np.array([[0.0]*n]*n)

            if distance_function == 'interpolated':
                D1, embeddings = self.get_distance_matrix(D1, embeddings=embeddings)
                D2, embeddings = self.get_distance_matrix(D2, embeddings=embeddings)

            elif distance_function == 'cosine':
                print('Loading word embeddings from {}'.format(embeddings))
                embeddings = loadVectors(embeddings)


            increment = 100/n 
            progress = 0.0
            origCheckPoint = 0.1
            checkPoint = origCheckPoint

            for i in range(n):

                progress += increment
                if progress > checkPoint:
                    print('{}% complete learning {} distance matrix'.format(round(progress, 3), distance_function))
                    checkPoint += origCheckPoint

                for j in range(i+1, n):
                    wf1 = self.ind2wf[i]
                    wf2 = self.ind2wf[j]
                    p1 = self.ind2lem[i]
                    p2 = self.ind2lem[j]

                    if p1 == p2:
                        distance = 1

                    else:

                        ### get pairwise distance
                        if distance_function == 'weightedLev':
                            distance = 1 - self.get_likelihood_cell_mate(p1, wf1, p2, wf2)
                        elif distance_function == 'lev':
                            distance = 1 - self.get_likelihood_cell_mate_control(p1, wf1, p2, wf2)
                        elif distance_function == 'cosine':
                            distance = cosDist(wf1, wf2, embeddings)
                        elif distance_function == 'interpolated':
                            distance = interpolate(D1[i][j], D2[i][j], alpha)
                        else:
                            print('UNSUPPORTED DISTANCE METRIC {}'.format(distance_function))
                            exit()

                    matrix[i][j] = matrix[j][i] = distance
            
            ### cache calculated distance matrix
            print('Caching calculated {} distance matrix'.format(distance_function))
            with open(filename, 'wb') as outfile:
                np.save(outfile, matrix)

        return matrix, embeddings


    def cluster_cell_medoids(self, distance_function, debug=False, embeddings=None, D1=None, D2=None, alpha=None):  # TO ADD: CHANGE K-MEDOID CLUSTERING OUTPUT TO A SOFTMAX DISTRIBUTION TO FACILITATE DOWNSTREAM CELL REASSIGNMENTS

        # get distance metrics
        print('Getting {} distance matrix'.format(distance_function))

        if distance_function == 'interpolated':
            dist_matrix, embeddings = self.get_distance_matrix('interpolated', D1=D1, D2=D2, alpha=alpha, embeddings=embeddings)
        elif distance_function == 'weightedLev':
            dist_matrix, embeddings = self.get_distance_matrix('weightedLev')
        elif distance_function == 'lev':
            dist_matrix, embeddings = self.get_distance_matrix('lev')
        elif distance_function == 'cosine':
            dist_matrix, embeddings = self.get_distance_matrix('cosine', embeddings=embeddings)
        else:
            print('UNSUPPORTED DISTANCE METRIC {}'.format(distance_function))
            exit()



        ### DEBUG: WLEV_DIST_MAT IS STILL TOO BIASED TOWARD SIMILAR BASES.. COULD OUTPUT POSTERIOR PROB AND THEN GO THROUGH ENFORCING NO OVERABUNDANCE.. OR COULD FIX THIS BY MAKING CELLMATE DECISIONS ON A PARADIGM-PARADIGM BASIS
        # k-medoid clustering of distance metrics
        print('K-medoid clustering {} distance matrix'.format(distance_function))
        M, self.medoid_ind = kMedoids(dist_matrix, len(self.total_coordinates))
        # medoid_ind is a dictionary from cluster labels to an array of member ind's
        # M is a list of medoid-center ind's with indeces corresponding to cluster labels

        cluster2wf = {}
        wf2cluster = {}
        for cluster in self.medoid_ind:
            if debug:
                print('Cluster {}'.format(str(cluster)))
            cluster2wf[cluster] = {}
            for ind in self.medoid_ind[cluster]:
                wf = self.ind2wf[ind]
                cluster2wf[cluster][wf] = True
                if wf not in wf2cluster:
                    wf2cluster[wf] = {}
                wf2cluster[wf][cluster] = True
                lem = self.ind2lem[ind]
                if debug:
                    print('\t{}\t{} <- {}'.format(str('\n\t'.join(list('  '.join(x) for x in self.GOLD_data[lem][wf]))), wf, lem))
        if debug:
            print('\n')

        cellMates = get_cellMates_final(cluster2wf, wf2cluster)

        return cellMates, dist_matrix, embeddings


    def swapMedoids(self, dist_matrix):

        ### look for the most dissonant medoids
            ## keep swapping from bottom until we get to one that can't be improved
        first_unfixed = -1
        while first_unfixed < len(self.most_dissonant_cells) -1:
            first_unfixed += 1
            improved = False
            foundImprovement = True
            while foundImprovement:
                cell1 = self.most_dissonant_cells[first_unfixed][1]
                diss1 = self.most_dissonant_cells[first_unfixed][0]
                medoid1 = self.cell_medoid[cell1]
                maxGains = 0
                bestSwap = None # [[cell1,cell2], HIDs, HCDs]
                for mdcInd in range(first_unfixed+1, len(self.most_dissonant_cells)):
                    c2 = self.most_dissonant_cells[mdcInd]
                    cell2 = c2[1]
                    diss2 = c2[0]
                    gains = sum([diss1, diss2])
                    # check if the swap will
                        # decrease both dissonances

                    HIDs, HCDs = self.get_hypothetical_ind_dissonance(cell1, cell2, dist_matrix)

                    gains -= sum([HCDs[cell2].cohesion, HCDs[cell1].cohesion])
                    if gains > maxGains:
                        # print('Found a new best swap for cell {}:  cell {}'.format(cell1, cell2))
                        maxGains = gains 
                        bestSwap = [[cell1, cell2], HIDs, HCDs, mdcInd]



                if bestSwap == None:
                    foundImprovement = False

                else:
                    ### swapping medoids requires switching:
                        # cell_ind, ind_cell, 
                        # cell_medoid, medoid_cell
                        # ind_dissonance, cell_dissonance
                        # most_dissonant_cells

                    improved = True
                    cell1 = bestSwap[0][0]
                    cell2 = bestSwap[0][1]
                    HIDs = bestSwap[1]
                    HCDs = bestSwap[2]
                    mdcInd = bestSwap[3]
                    medoid1 = self.cell_medoid[cell1]
                    medoid2 = self.cell_medoid[cell2]
                    print('Swapping medoids at {}; {} (-{}% absolute dissonance)'.format(cell1, cell2, str(round(100*maxGains, 3))))
                    # cell dissonance
                    self.cell_dissonance[cell1] = HCDs[cell1]
                    self.cell_dissonance[cell2] = HCDs[cell2]
                    # most dissonant cells
                    self.most_dissonant_cells[first_unfixed] = [HCDs[cell1].cohesion, cell1]
                    self.most_dissonant_cells[mdcInd] = [HCDs[cell2].cohesion, cell2]

                    for ind in HIDs:
                        # get original and new cells
                        origCell = self.ind_cell[ind]
                        if origCell == cell1:
                            cell = cell2 
                        else:
                            assert origCell == cell2 
                            cell = cell1
                        medoid = self.ind_medoid[ind]

                        # ind cell mapping
                        self.ind_cell[ind] = cell
                        del self.cell_ind[origCell][ind]
                        self.cell_ind[cell][ind] = True
                        # medoid cell mapping
                        self.medoid_cell[medoid] = cell
                        self.cell_medoid[cell] = medoid
                        # ind dissonance
                        self.ind_dissonance[ind] = HIDs[ind]




            print('Fixed {} of {} medoids in their respective cells'.format(str(first_unfixed+1), str(len(self.most_dissonant_cells))))
            coordinates = self.most_dissonant_cells[first_unfixed-1][1]
            fts = get_MSP_from_coord(self.GOLD_skeleton, coordinates)
            print('\tNewest fixed cell: {}'.format(fts)) # == {}'.format(coordinates, fts))

            if first_unfixed < len(self.most_dissonant_cells) -1 and improved:
                self.get_ind_dissonance(dist_matrix)

        return self.get_cellMates_final()


    def cell_mapping_and_word_reassignment(self, dist_matrix):

        self.assignMedoid_random(dist_matrix)

        ### debug secondary eval cell prediction
        ana.get_cell_wf()
        cellF_rand = ana.evalFull()

        return self.swapMedoids(dist_matrix)
        ### debug secondary eval cell prediction
        ana.get_cell_wf()
        cellF_rand = ana.evalFull()


        ### when medoids swap
            ## cohesion and silhoette remain unchanged
            ## all ind_fitnesses need to be recalculated
        ### when inds get reassigned
            ## reaveraging of medoid cohesion is trivial
            ## reaveraging of cell_fitness is trivial if
                # we ignore the affects on ind2s in m of moving ind1 to related cell
            ## ind_silhoette must be recalculated in new cell for ind1
            ## ind_silhoette of ind2s should be adjusted to account for ind1s absence
            ## ind_fitness must be recalculated for ind1 in new cell
            ## ind_fitness can be trivially ignored if
                # we ignore affects on ind2s in m of moving ind1 to related cell

        ### To make it tractable and to introduce some potentially helpful noise, we can sample to calculate fitness and silhoettes, recalculating after each epoch

        ### medoids will be swapped when it improves cell_fitness for both

        ### inds will be reassigned when ind_fitness improvement outweighs the loss in medoid cohesion

        ### ind_fitness should be a function of the proportion of members whose nearest medoid occupies a feature sharing cell 


    def assignMedoid_random(self, dist_matrix):

        self.cell_ind = {}
        self.ind_cell = {}
        self.feat_cell = {}
        self.medoid_cell = {}
        self.cell_medoid = {}
        self.ind_medoid = {}

        random_cells = list(self.total_coordinates)
        random.shuffle(random_cells)

        for medoid in self.medoid_ind:
            cell = random_cells.pop()
            fts = cell.split(',')
            fts = list('{},{}'.format(str(i), fts[i]) for i in range(len(fts)))
            for ft in fts:
                if ft not in self.feat_cell:
                    self.feat_cell[ft] = {}
                self.feat_cell[ft][cell] = True

            for ind in self.medoid_ind[medoid]:

                self.medoid_cell[medoid] = cell
                self.ind_medoid[ind] = medoid
                self.cell_medoid[cell] = medoid
                self.ind_cell[ind] = cell
                if cell not in self.cell_ind:
                    self.cell_ind[cell] = {}
                self.cell_ind[cell][ind] = True

        assert len(random_cells) == 0

        ### calculate index silhoettes
        self.get_ind_silhoette(dist_matrix)
        ### calculate medoid cohesion
        self.get_medoid_cohesion()
        ### calculate ind dissonance
        self.get_ind_dissonance(dist_matrix)
        ### calculate cell dissonance
        self.get_cell_dissonance()


    def get_ind_silhoette(self, dist_matrix):
        print('Getting index silhoettes')
        label_vector = np.array([0]*len(dist_matrix))
        for label in self.medoid_ind:
            for ind in self.medoid_ind[label]:
                label_vector[ind] = label
        self.ind_silhoette = silhouette_samples(dist_matrix, label_vector, metric='precomputed')

        self.least_cohesive_inds = []
        for ind in range(len(self.ind_silhoette)):
            self.least_cohesive_inds.append([self.ind_silhoette[ind], ind])
        self.least_cohesive_inds.sort()


    def get_medoid_cohesion(self):
        print('Calculating medoid cohesion')
        self.medoid_cohesion = {}  # average of composite silhoettes
        for ind in range(len(self.ind_silhoette)):
            medoid = self.ind_medoid[ind]
            if medoid not in self.medoid_cohesion:
                self.medoid_cohesion[medoid] = fitness_tracker()
                self.medoid_cohesion[medoid].addSil(self.ind_silhoette[ind])
        for medoid in self.medoid_cohesion:
            self.medoid_cohesion[medoid].normSil()

        self.least_cohesive_medoids = []
        for medoid in self.medoid_cohesion:
            self.least_cohesive_medoids.append([self.medoid_cohesion[medoid].cohesion, medoid])
        self.least_cohesive_medoids.sort()


    def get_hypothetical_ind_dissonance(self, cell1, cell2, dist_matrix):

        hypothetical_ind_dissonance = {}
        hypothetical_cell_dissonance = {}

        for ind1 in self.cell_ind[cell1]:
            hypothetical_ind_dissonance[ind1] = fitness_tracker()
        for ind2 in self.cell_ind[cell2]:
            hypothetical_ind_dissonance[ind2] = fitness_tracker()
        newCell1 = cell2 
        newCell2 = cell1
        hypothetical_cell_dissonance[newCell1] = fitness_tracker()
        hypothetical_cell_dissonance[newCell2] = fitness_tracker()

        fts1 = set(self.strCoord_lstCoord[newCell1])
        fts2 = set(self.strCoord_lstCoord[newCell2])
        multiplier = len(fts1.intersection(fts2))
        if multiplier > 0:
            for ind1 in self.cell_ind[cell1]:
                for ind2 in self.cell_ind[cell2]:
                    hypothetical_ind_dissonance[ind1].multSil(multiplier, dist_matrix[ind1][ind2])
                    hypothetical_ind_dissonance[ind2].multSil(multiplier, dist_matrix[ind1][ind2])

        for cell3 in self.cell_ind:
            if cell3 not in (newCell1, newCell2):
                fts3 = self.strCoord_lstCoord[cell3]

                multiplier1 = len(fts1.intersection(fts3))
                if multiplier1 > 0:
                    for ind1 in self.cell_ind[cell1]:
                        for ind3 in self.cell_ind[cell3]:
                            hypothetical_ind_dissonance[ind1].multSil(multiplier, dist_matrix[ind1][ind3])

                multiplier2 = len(fts2.intersection(fts3))
                if multiplier2 > 0:
                    for ind2 in self.cell_ind[cell2]:
                        for ind3 in self.cell_ind[cell3]:
                            hypothetical_ind_dissonance[ind2].multSil(multiplier, dist_matrix[ind2][ind3])


        for ind1 in self.cell_ind[cell1]:
            hypothetical_ind_dissonance[ind1].normSil()
            hypothetical_cell_dissonance[newCell1].addSil(hypothetical_ind_dissonance[ind1].cohesion)
        hypothetical_cell_dissonance[newCell1].normSil()

        for ind2 in self.cell_ind[cell2]:
            hypothetical_ind_dissonance[ind2].normSil()
            hypothetical_cell_dissonance[newCell2].addSil(hypothetical_ind_dissonance[ind2].cohesion)
        hypothetical_cell_dissonance[newCell2].normSil()

        return hypothetical_ind_dissonance, hypothetical_cell_dissonance


    def get_ind_dissonance(self, dist_matrix):


        print('Calculating index dissonance')
        self.ind_dissonance = {}  # how well ind fits with related cells

        for ind in range(len(dist_matrix)):
            self.ind_dissonance[ind] = fitness_tracker()

        all_cells = list(self.cell_ind)
        all_cells.sort()
        for c1 in range(len(all_cells)):
            cell1 = all_cells[c1]
            fts1 = set(self.strCoord_lstCoord[cell1])
            for c2 in range(c1+1, len(all_cells)):
                cell2 = all_cells[c2]
                fts2 = set(self.strCoord_lstCoord[cell2])
                multiplier = len(fts1.intersection(fts2))
                if multiplier > 0:
                    for ind1 in self.cell_ind[cell1]:
                        for ind2 in self.cell_ind[cell2]:
                            self.ind_dissonance[ind1].multSil(multiplier, dist_matrix[ind1][ind2])
                            self.ind_dissonance[ind2].multSil(multiplier, dist_matrix[ind1][ind2])

        for ind in range(len(dist_matrix)):
            self.ind_dissonance[ind].normSil()

        self.most_dissonant_inds = []
        for ind in self.ind_dissonance:
            self.most_dissonant_inds.append([self.ind_dissonance[ind].cohesion, ind])
        self.most_dissonant_inds.sort(reverse=True)


    def get_cell_dissonance(self):
        print('Calculating cell dissonance')
        self.cell_dissonance = {}  # average of composite fitness values
        for ind in self.ind_dissonance:
            self.ind_dissonance[ind].normSil()
            cell = self.ind_cell[ind]
            if cell not in self.cell_dissonance:
                self.cell_dissonance[cell] = fitness_tracker()
            self.cell_dissonance[cell].addSil(self.ind_dissonance[ind].cohesion)
        for cell in self.cell_dissonance:
            self.cell_dissonance[cell].normSil()

        self.most_dissonant_cells = []
        for cell in self.cell_dissonance:
            self.most_dissonant_cells.append([self.cell_dissonance[cell].cohesion, cell])
        self.most_dissonant_cells.sort(reverse=True)


    def get_cell_wf(self):

        self.cell_wf = {}

        for cell in self.cell_ind:
            if cell not in self.cell_wf:
                self.cell_wf[cell] = {}
            for ind in self.cell_ind[cell]:
                wf = self.ind2wf[ind]
                self.cell_wf[cell][wf] = True


    def evalFull(self):

        numer = 0
        denom = 0
        minF = 101
        maxF = -1

        for cell in self.GOLD_cell_wf:

            predicted = set(self.cell_wf[cell])
            actual = set(self.GOLD_cell_wf[cell])
            correct = predicted.intersection(actual)

            if len(correct) == 0:
                F = 0
            else:
                prec = len(correct) / len(predicted)
                rec = len(correct) / len(actual)
                F = 100 * ( (2 * prec * rec) / (prec + rec) )
                
            weight = len(actual)

            if F < minF:
                minF = F 
            if F > maxF:
                maxF = F

            numer += weight * F 
            denom += weight 

        macroF = numer / denom
        strMacroF = str(round(macroF, 3))
        strMinF = str(round(minF, 3))
        strMaxF = str(round(maxF, 3))

        print('\n\nMacro F: {}\t(min: {}, max: {}, total cells: {})'.format(strMacroF, strMinF, strMaxF, str(len(self.cell_wf))))

        return macroF


    def get_cellMates_final(self):

        cellMates = {wf:{} for wf in self.wf2ind}

        increment = 100/len(self.cell_ind)
        progress = 0
        nextTen = 10

        print('Getting cellMates from an array')
        for cell in self.cell_ind:
            progress += increment

            to_add = {}
            for ind in self.cell_ind[cell]:
                to_add[self.ind2wf[ind]] = True

            wfs = dict(to_add)
            for wf1 in wfs:
                del to_add[wf1]
                for wf2 in to_add:
                    cellMates[wf1][wf2] = True 
                    cellMates[wf2][wf1] = True
            if progress > nextTen:
                print('\t{}% complete getting cellMates'.format(round(progress, 3)))
                nextTen += 10

            for wf in cellMates:
                try:
                    del cellMates[wf][wf]
                except KeyError:
                    pass

        return cellMates


###################################################################################
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def interpolate(v1, v2, alpha):
    value = alpha*v1 + (1-alpha)*v2
    return value

def print_sample_paradigms(ana, limit, output):

    output = open(output, 'w')

    randLems = list(ana.GOLD_data)
    random.shuffle(randLems)

    try:
        limit = int(limit)

        for lem in randLems:
            for wf in ana.GOLD_data[lem]:
                for fts in ana.GOLD_data[lem][wf]:
                    output.write('{}\t{}\t{}\n'.format(lem, wf, ';'.join(fts)))
                    limit -= 1
            if limit < 1:
                break

    except:

        for lem in randLems:
            for wf in ana.GOLD_data[lem]:
                for fts in ana.GOLD_data[lem][wf]:
                    if limit in fts:
                        output.write('{}\t{}\t{}\n'.format(lem, wf, ';'.join(fts)))

    output.close()

    exit()

def eval(GOLD_cellMates, cellMates, ana, debug=False):

    print('EVALUATING... (THIS MAY TAKE A HOT SEC)')
    total = len(GOLD_cellMates)
    increment = 100/total

    # take macro F score of cellMates over all forms in gold vocabulary
    Fs = []
    minF = 2
    maxF = -1
    counter = 0
    nextTen = 10
    for wf in GOLD_cellMates:
 
        try:
            correctSet = set(GOLD_cellMates[wf]).intersection(set(cellMates[wf]))
            correct = len(correctSet)
        except IndexError:
            correctSet = set()
            correct = 0

        if correct == 0:
            F = 0
        else:
            predicted = len(cellMates[wf])
            total_recall_with_Arnold_Schwarzeneggar = len(GOLD_cellMates[wf])
            
            prec = correct / predicted
            rec = correct / total_recall_with_Arnold_Schwarzeneggar
            F = 100*( (2 * prec * rec) / (prec + rec) )

        if F < minF:
            minF = F 
        if F > maxF:
            maxF = F
        Fs.append(F)

        if debug:

            correctFeats = {}
            print('\n\n{}\t{} F-score'.format(wf, str(round(F, 2))))
            for par in ana.wf2lem[wf]:
                print('\tLemma {}'.format(par))
                for fts in ana.GOLD_data[par][wf]:
                    print('\t\t{}'.format(';'.join(fts)))
                    correctFeats[';'.join(fts)] = True

            precErrors = {}
            precErrorFts = {}
            tpe = 0
            tpef = 0
            for x in cellMates[wf]:
                assert x in GOLD_cellMates
                if x not in GOLD_cellMates[wf]:
                    precErrors[x] = True
                    tpe += 1
                    assert len(ana.wf2lem[x]) > 0
                    for par in ana.wf2lem[x]:
                        assert len(ana.GOLD_data[par][x]) > 0
                        for fts in ana.GOLD_data[par][x]:
                            fts = ';'.join(fts)
                            if fts not in precErrorFts:
                                precErrorFts[fts] = 0
                            precErrorFts[fts] += 1
                            tpef += 1
            assert tpe <= tpef

            tre = 0
            tref = 0
            recErrors = {}
            recErrorFts = {}
            for x in GOLD_cellMates[wf]:
                assert x in cellMates
                if x not in cellMates[wf]:
                    recErrors[x] = True
                    tre += 1
                    for par in ana.wf2lem[x]:
                        for fts in ana.GOLD_data[par][x]:
                            fts = ';'.join(fts)
                            if fts not in recErrorFts:
                                recErrorFts[fts] = 0
                            recErrorFts[fts] += 1
                            tref += 1
            assert tre <= tref

            if len(correctSet) == 0:
                assert F == 0
            else:
                assert len(correctSet) + len(precErrors) == predicted
                assert len(correctSet) + len(recErrors) == total_recall_with_Arnold_Schwarzeneggar

            assert len(precErrorFts.values()) == len(precErrorFts.keys())
            assert len(recErrorFts.values()) == len(recErrorFts.keys())

            rankedPrecErrorFts = list(zip(list(precErrorFts.values()), list(precErrorFts.keys())))
            rankedPrecErrorFts.sort(reverse=True)
            rankedPrecErrorFts = '  '.join('{}_({})'.format(l[1], str(l[0])) for l in rankedPrecErrorFts)

            rankedRecErrorFts = list(zip(list(recErrorFts.values()), list(recErrorFts.keys())))
            rankedRecErrorFts.sort(reverse=True)
            rankedRecErrorFts = '  '.join('{}_({})'.format(l[1], str(l[0])) for l in rankedRecErrorFts)

            print('\tCORRECT:\n{}'.format(', '.join(list(correctSet))))
            print('\tPREC ERRORS:\n{}\n{}'.format(rankedPrecErrorFts, ', '.join(list(precErrors))))
            print('\tREC ERRORS:\n{}\n{}'.format(rankedRecErrorFts, ', '.join(list(recErrors))))

        counter += increment
        if counter > nextTen:
            print('\t{}% complete evaluating cellMate matches'.format(str(round(counter, 0))))
            nextTen += 10

    macroF = sum(Fs)/len(Fs)
    strMacroF = str(round(macroF, 2))
    strMinF = str(round(minF, 2))
    strMaxF = str(round(maxF, 2))
    stdev = statistics.stdev(Fs)
    strStdev = str(round(stdev, 2))

    print('\n\nMacro F: {}\t(min: {}, max: {}, stdev: {}, total instances: {})'.format(strMacroF, strMinF, strMaxF, strStdev, str(total)))

    return macroF

def get_cellMates(array):

    ### learn mapping from coordinates to forms and initialize cellMates dictionary
    coordinates_to_forms = {}
    forms_to_coordinates = {}

    for lemma in array:
        for wf in array[lemma]:
            if wf not in forms_to_coordinates:
                forms_to_coordinates[wf] = {}
            for coord in array[lemma][wf]:
                coord = ','.join(str(x) for x in coord)
                forms_to_coordinates[wf][coord] = True
                if coord not in coordinates_to_forms:
                    coordinates_to_forms[coord] = {}
                coordinates_to_forms[coord][wf] = True

    cellMates = get_cellMates_final(coordinates_to_forms, forms_to_coordinates)

    return cellMates, coordinates_to_forms, forms_to_coordinates

def get_cellMates_final(cluster2wf, wf2cluster):

    ### learn cellMates from coordinate-form mappings
    cellMates = {wf:{} for wf in wf2cluster}

    increment = 100/len(cluster2wf)
    progress = 0
    nextTen = 10

    print('Getting cellMates from an array')
    for coord in cluster2wf:
        progress += increment
        ### DEBUG COMPLEXITY ISSUES
        # print('\t\t{} cell mates at {}'.format(str(len(cluster2wf[coord])**2 - len(cluster2wf[coord])), coord))

        to_add = dict(cluster2wf[coord])
        for wf1 in cluster2wf[coord]:
            del to_add[wf1]
            for wf2 in to_add:
                cellMates[wf1][wf2] = True 
                cellMates[wf2][wf1] = True
        if progress > nextTen:
            print('\t{}% complete getting cellMates'.format(round(progress, 3)))
            nextTen += 10

        for wf in cellMates:
            try:
                del cellMates[wf][wf]
            except KeyError:
                pass

    return cellMates

class fitness_tracker:

    def __init__(self):

        self.size = 0
        self.sum_sil = 0

    def addSil(self, sil):

        self.size += 1
        self.sum_sil += sil

    def multSil(self, mult, sil):

        self.size += mult
        self.sum_sil += sil*mult

    def divSil(self, quot, sil):

        self.size -= quot
        self.sum_sil -= sil*quot

    def normSil(self):

        self.cohesion = self.sum_sil / self.size 

    def subSil(self, sil):

        self.size -= 1
        self.sum_sil -= sil

class continuous_range(object):

    def __init__(self, minV, maxV):
        self.min = minV
        self.max = maxV
    def __eq__(self, v):
        return self.min <= v <= self.max

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_MSP_from_coord(skeleton, coord):

        ftList = []
        coord = list(int(x) for x in coord.split(','))
        for dimInd in range(len(coord)):
            featInd = coord[dimInd]
            if featInd != 0:
                ftList.append(skeleton[dimInd][featInd])
        return ';'.join(ftList)


###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--unimorph', type=str, help='Location of Unimorph file', required=True)
    parser.add_argument('-e', '--embeddings', type=str, help='Location of word embeddings', required=False, default=None)
    parser.add_argument('-l', '--limit', type=str, help='Shall we draw a sample of the full unimorph data? You can limit it by number of instances or by shared feature.', required=False, default=None)
    parser.add_argument('-d', '--distance_function', type=str, choices=['interpolated', 'weightedLev', 'lev', 'cosine', 'random'], help='Metric that determines the relevant distance matrix used for cell clustering', required=False, default='interpolated')
    parser.add_argument('-x', '--d1', type=str, choices=['weightedLev', 'lev', 'cosine'], help='First metric considered in the interpolated distance function', required=False, default='weightedLev')
    parser.add_argument('-y', '--d2', type=str, choices=['weightedLev', 'lev', 'cosine'], help='Second metric considered in the interpolated distance function', required=False, default='cosine')
    parser.add_argument('-a', '--alpha', type=float, choices=[continuous_range(0.0, 1.0)], help='Relative weight placed on d1 for the interpolated distance metric', required=False, default=0.5)
    parser.add_argument('-m', '--debugMedoids', type=str2bool, help='Show medoids during clustering for debugging/development purposes', required=False, default=False)
    parser.add_argument('-c', '--debugCells', type=str2bool, help='Show cells during clustering for debugging/development purposes', required=False, default=False)
    args = parser.parse_args()


    UNIMORPH_LG = args.unimorph

    ########################### READ IN DATA ######################################
    ### read in file, get UG, Gold, plain (Gold without tags) data/array/skeletons
    ana = ANA(UNIMORPH_LG)
    ### generate a sample file if requested
    if args.limit != None:
        print_sample_paradigms(ana, args.limit, '{}.sample-{}'.format(UNIMORPH_LG, args.limit))

    ## get all possible array coordinates and word classes according to Gold
    ana.get_attested_coordinates()    # in ara, 196 attested coordinates, 19 word classes
                                    # in deu,  37 attested coordinates,  9 word classes

    ########################### INITIALIZATION ######################################
    ### k-medoid cluster cells based on these (potentially interpolated) metrics:
        # random, weighted Lev ED, Lev ED, and/or cosine similarity
    ana.learn_exponence_weights()
    if args.distance_function == 'random':
        cellMates = ana.assignCellMates_random()
    else:
        cellMates, dist_matrix, args.embeddings = ana.cluster_cell_medoids(args.distance_function, D1=args.d1, D2=args.d2, embeddings=args.embeddings, debug=args.debugMedoids, alpha=args.alpha) # debug will show cell medoids

    ################## MEDOID TO CELL MAPPING AND WF RE-ASSIGNMENT ##################

    cellMates = ana.cell_mapping_and_word_reassignment(dist_matrix)

    ############################### EVALUATION ######################################
    GOLD_cellMates, GOLD_coordinates_to_forms, GOLD_forms_to_coordinates = get_cellMates(ana.GOLD_array) # pickle these
    macroF = eval(GOLD_cellMates, cellMates, ana, debug=args.debugCells) # debug shows wf's F,p,r, and cellMates

    ### Secondary evaluation on cell prediction
    # ana.get_cell_wf()
    # cellF_swappedMedoids = ana.evalFull()




    

