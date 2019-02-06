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
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine


ANA_DIR = os.path.dirname(os.path.realpath(argv[0]))
MAPPING_DIR = os.path.join(ANA_DIR, 'Mappings')
UNIMORPH_UG = os.path.join(ANA_DIR, 'Unimorph/unimorph_feature_dimensions.tsv')


###################################################################################
class ANA:

    def __init__(self, fn):
        

        ### get UG dimensions of features
        self.UG_dim_feats = {}
        self.UG_feats_dim = {}

        for line in open(UNIMORPH_UG):
            dim, ft = line.split()
            ft = ft.upper()
            if dim not in self.UG_dim_feats:
                self.UG_dim_feats[dim] = {}
            self.UG_dim_feats[dim][ft] = True
            if ft not in self.UG_feats_dim:
                self.UG_feats_dim[ft] = dim


        ### get gold data, feat-dim maps, no-fts data, dimension order
        print('Reading in gold data and building maps..')
        self.GOLD_lem_wf_lstFtlists = {}
        self.dimOrder = []
        self.lem_wf = {}
        self.GOLD_dim_feats = {}
        self.GOLD_feats_dim = {}

        read_lines = 0
        for line in open(fn):
            line = line.strip('\n')
            if line:
                line = line.split('\t')
                lemma, wf, fts = map(str, line)

                ## record gold and featureless data
                if lemma not in self.GOLD_lem_wf_lstFtlists:
                    self.GOLD_lem_wf_lstFtlists[lemma] = {}
                    self.lem_wf[lemma] = {}
                if wf not in self.GOLD_lem_wf_lstFtlists[lemma]:
                    self.GOLD_lem_wf_lstFtlists[lemma][wf] = []
                    self.lem_wf[lemma][wf] = True

                ## handle feature notation inconsistencies
                fts = fts.replace('NDEF','INDF')
                fts = fts.split(';')
                # check for multiple feats per dimension
                type_dim_fts = {}
                to_remove = {}
                for ft in fts:
                    dim = self.UG_feats_dim[ft]
                    if dim not in type_dim_fts:
                        type_dim_fts[dim] = []
                    type_dim_fts[dim].append(ft)
                    if len(type_dim_fts[dim]) > 1:
                        assert len(type_dim_fts[dim]) == 2
                        prefix = type_dim_fts[dim][0].split('.')[0]
                        assert prefix == type_dim_fts[dim][1].split('.')[0]
                        to_remove[prefix] = True
                if len(to_remove) > 0:
                    new_fts = []
                    for ft in fts:
                        if ft not in to_remove:
                            new_fts.append(ft)
                    fts = new_fts
                self.GOLD_lem_wf_lstFtlists[lemma][wf].append(fts)

                ## induce preferred order of feature dimensions
                lowestIndex = 0
                for ft in fts:
                    dim = self.UG_feats_dim[ft]
                    if dim not in self.GOLD_dim_feats:
                        self.GOLD_dim_feats[dim] = {}
                    if ft not in self.GOLD_dim_feats[dim]:
                        self.GOLD_dim_feats[dim][ft] = True
                        self.GOLD_feats_dim[ft] = dim
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
        assert len(self.dimOrder) == len(self.GOLD_dim_feats)


        ### get UG skeleton and UG feat to skeleton map
        self.UG_feats_to_skel_coordinates = {}
        self.UG_skeleton = []

        for dimInd in range(len(self.dimOrder)):
            dim = self.dimOrder[dimInd]
            featInd = 0
            self.UG_skeleton.append(['NONE'])
            for ft in self.UG_dim_feats[dim]:
                featInd += 1
                self.UG_skeleton[-1].append(ft)
                self.UG_feats_to_skel_coordinates[ft] = [dimInd, featInd]


        ### get gold skeleton and feature map with consistent ordering
        ## limit gold skeleton to only UG features relevant in this lg
        self.GOLD_skeleton = []
        self.GOLD_fts_skel_coordinates = {}

        for dimInd in range(len(self.dimOrder)):
            dim = self.dimOrder[dimInd]
            self.GOLD_skeleton.append(['NONE'])
            featInd = 0
            for ft in self.GOLD_dim_feats[dim]:
                self.GOLD_skeleton[-1].append(ft)
                featInd += 1
                self.GOLD_fts_skel_coordinates[ft] = [dimInd, featInd]


        ### get gold array of paradigms
        print('Mapping paradigms to gold cells..')
        self.GOLD_lem_wf_lstSkelCoordlsts = {}

        for lemma in self.GOLD_lem_wf_lstFtlists:
            self.GOLD_lem_wf_lstSkelCoordlsts[lemma] = {}
            for wf in self.GOLD_lem_wf_lstFtlists[lemma]:
                self.GOLD_lem_wf_lstSkelCoordlsts[lemma][wf] = []
                for fts in self.GOLD_lem_wf_lstFtlists[lemma][wf]:
                    self.GOLD_lem_wf_lstSkelCoordlsts[lemma][wf].append([0] * len(self.GOLD_skeleton))
                    for ft in fts:
                        coord = self.GOLD_fts_skel_coordinates[ft]
                        assert ft == self.GOLD_skeleton[coord[0]][coord[1]]
                        self.GOLD_lem_wf_lstSkelCoordlsts[lemma][wf][-1][coord[0]] = coord[1]


        ## integerize everything
        print('Integerizing data..')
        self.wf_lem = {}
        self.ind_wf = {}
        self.wf_ind = {}
        self.lem_ind = {}
        self.ind_lem = {}

        ind = -1
        deterministicLemOrder = list(self.lem_wf)
        deterministicLemOrder.sort()
        for lem in deterministicLemOrder:
            deterministicWForder = list(self.lem_wf[lem])
            deterministicWForder.sort()
            for wf in deterministicWForder:
                ind += 1
                ## manage wf's
                if wf not in self.wf_lem:
                    self.wf_lem[wf] = {}
                self.wf_lem[wf][lem] = True
                if wf not in self.wf_ind:
                    self.wf_ind[wf] = {}
                self.wf_ind[wf][ind] = True
                ## manage lemmas's
                if lem not in self.lem_ind:
                    self.lem_ind[lem] = {}
                self.lem_ind[lem][ind] = True
                ## manage ind's
                self.ind_wf[ind] = wf
                self.ind_lem[ind] = lem

    def get_attested_cells(self):

        print('Attesting cells..')
        ### get all attested coordinates and word classes
        self.allCells = {}
        lemma_coords = {}
        self.metaParadigms = {}
        self.cell_lstCoord = {}
        self.GOLD_cell_wf = {}
        self.GOLD_wf_cell = {}

        for lemma in self.GOLD_lem_wf_lstSkelCoordlsts:
            if lemma not in lemma_coords:
                lemma_coords[lemma] = {}
            for wf in self.GOLD_lem_wf_lstSkelCoordlsts[lemma]:
                for realization in self.GOLD_lem_wf_lstSkelCoordlsts[lemma][wf]:
                    cell = ','.join(str(x) for x in realization)

                    if cell not in self.GOLD_cell_wf:
                        self.GOLD_cell_wf[cell] = {}
                    self.GOLD_cell_wf[cell][wf] = True

                    if wf not in self.GOLD_wf_cell:
                        self.GOLD_wf_cell[wf] = {}
                    self.GOLD_wf_cell[wf][cell] = True

                    self.cell_lstCoord[cell] = []
                    for f in range(len(realization)):
                        self.cell_lstCoord[cell].append('{},{}'.format(f, realization[f]))

                    self.allCells[cell] = True
                    lemma_coords[lemma][cell] = True
            all_coords = list(lemma_coords[lemma].keys())
            all_coords.sort()
            all_coords = '\n'.join(all_coords)
            self.metaParadigms[all_coords] = True

        print('Mapping cells to possible word classes and paradigm mate cells')
        ### map all coordinates to possible wordClasses and possible paradigm mates
        self.cell_metaParadigm = {}
        self.metaParadigm_cell = {}
        self.cell_metaParadigmMate = {}

        for wc in self.metaParadigms:
            self.metaParadigm_cell[wc] = {}
            wcList = wc.split('\n')
            for c1 in range(len(wcList)):
                cell1 = wcList[c1]
                if cell1 not in self.cell_metaParadigm:
                    self.cell_metaParadigm[cell1] = {}
                self.cell_metaParadigm[cell1][wc] = True
                self.metaParadigm_cell[wc][cell1] = True

                for c2 in range(c1+1, len(wcList)):
                    cell2 = wcList[c2]
                    if c1 not in self.cell_metaParadigmMate:
                        self.cell_metaParadigmMate[c1] = {}
                    if c2 not in self.cell_metaParadigmMate:
                        self.cell_metaParadigmMate[c2] = {}
                    self.cell_metaParadigmMate[c1][c2] = True
                    self.cell_metaParadigmMate[c2][c1] = True

        self.get_GOLD_cellMates()


    def learn_exponence_weights(self, bias_against_overabundance=1):  ### TO ADD: LINEAR INTERPOLATION WITH NGRAM EDIT BLOCKS

        try:
            if bias_against_overabundance == 1:
                self.condExpWeight = pkl.load(open('{}.condExpWeight.pkl'.format(UNIMORPH_LG), 'rb'))
            else:
                self.condExpWeight = pkl.load(open('{}-{}.condExpWeight.pkl'.format(UNIMORPH_LG, str(bias_against_overabundance)), 'rb'))
            print('Reading cached paradigm conditional exponence weights')
            self.baseWeights = pkl.load(open('{}.baseWeights.pkl'.format(UNIMORPH_LG), 'rb'))
            self.exponenceWeights = pkl.load(open('{}.exponenceWeights.pkl'.format(UNIMORPH_LG), 'rb'))

        except FileNotFoundError:
            print('Learning paradigm conditional exponence weights')
            self.baseWeights = {}
            self.exponenceWeights = {}
            self.condExpWeight = {}

            ### get edit distance between all possible paradigm mate pairs
            counter = 0
            for lemma in self.lem_wf:
                self.baseWeights[lemma] = {}

                eligibleWF2 = dict(self.lem_wf[lemma])
                for wf1 in self.lem_wf[lemma]:
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
                    assert len(self.lem_wf[lemma]) == 1
                    for wf in self.lem_wf[lemma]:
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
                    self.condExpWeight[lemma][ch] = self.exponenceWeights[ch] / (self.exponenceWeights[ch] + bias_against_overabundance*self.baseWeights[lemma][ch])

            ### cache learned weights
            pkl.dump( self.baseWeights, open('{}.baseWeights.pkl'.format(UNIMORPH_LG), 'wb' ) )
            pkl.dump( self.exponenceWeights, open('{}.exponenceWeights.pkl'.format(UNIMORPH_LG), 'wb' ) )
            if bias_against_overabundance == 1:
                pkl.dump( self.condExpWeight, open('{}.condExpWeight.pkl'.format(UNIMORPH_LG), 'wb' ) )
            else:
                pkl.dump( self.condExpWeight, open('{}-{}.condExpWeight.pkl'.format(UNIMORPH_LG, str(bias_against_overabundance)), 'wb' ) )


    def assignCells_random(self):

        print('Randomly assigning forms to cells')

        self.cell_wf = {}
        self.wf_cell = {}
        self.ind_cell = {}
        self.cell_ind = {}
        
        for lem in self.lem_wf:
            self.assignParadigm_random(lem)

        self.get_cellMates()


    def assignParadigm_random(self, lem):

        random_cells = random.sample(list(self.allCells), len(self.lem_ind[lem]))
        random.shuffle(random_cells)

        for ind in self.lem_ind[lem]:
            cell = random_cells.pop()
            wf = self.ind_wf[ind]

            # update wf_cell
            if wf not in self.wf_cell:
                self.wf_cell[wf] = {}
            self.wf_cell[wf][cell] = True
            # update cell_wf and cell_ind
            if cell not in self.cell_wf:
                self.cell_wf[cell] = {}
                self.cell_ind[cell] = {}
            self.cell_wf[cell][wf] = True
            self.cell_ind[cell][ind] = True
            # update ind_cell
            self.ind_cell[ind] = cell


    def get_cellMates(self):

        # Initialize bidirectional cellMates dictionary
        self.cellMates = {}

        # Keep track of word forms we've already accounted for
        eligibleWFs = dict(self.wf_cell)

        # For each wf, make sure it has a key in the cellMates dictionary
        for wf1 in self.wf_cell:
            if wf1 not in self.cellMates:
                self.cellMates[wf1] = {}
            # Mark that it won't have to be considered in the future
            del eligibleWFs[wf1]

            # Check what other wf's share a cell with this wf
            cell1s = set(self.wf_cell[wf1])
            for wf2 in eligibleWFs:
                for cell2 in self.wf_cell[wf2]:

                    # Once you find a match, record in both directions and break
                    if cell2 in cell1s:
                        self.cellMates[wf1][wf2] = True 

                        if wf2 not in self.cellMates:
                            self.cellMates[wf2] = {}
                        self.cellMates[wf2][wf1] = True

                        break


    def get_GOLD_cellMates(self):

        # ### WF1 by WF2 method

        # print('Getting gold cell mates')
        # # Initialize bidirectional cellMates dictionary
        # self.GOLD_cellMates = {}

        # # Keep track of word forms we've already accounted for
        # eligibleWFs = dict(self.GOLD_wf_cell)

        # # For each wf, make sure it has a key in the cellMates dictionary
        # for wf1 in self.GOLD_wf_cell:


        #     print(len(eligibleWFs))


        #     if wf1 not in self.GOLD_cellMates:
        #         self.GOLD_cellMates[wf1] = {}
        #     # Mark that it won't have to be considered in the future
        #     del eligibleWFs[wf1]

        #     # Check what other wf's share a cell with this wf
        #     cell1s = set(self.GOLD_wf_cell[wf1])
        #     for wf2 in eligibleWFs:
        #         for cell2 in self.GOLD_wf_cell[wf2]:

        #             # Once you find a match, record in both directions and break
        #             if cell2 in cell1s:
        #                 self.GOLD_cellMates[wf1][wf2] = True 

        #                 if wf2 not in self.GOLD_cellMates:
        #                     self.GOLD_cellMates[wf2] = {}
        #                 self.GOLD_cellMates[wf2][wf1] = True

        #                 break



        ###################

        ### Cell method

        print('Getting gold cell mates')
        # Initialize bidirectional cellMates dictionary
        self.GOLD_cellMates = {}

        progress = 0
        increment = 100/len(self.GOLD_cell_wf)

        for cell in self.GOLD_cell_wf:
            progress += increment
            print('\t{}% complete'.format(str(round(progress, 2))))
            eligibleWFs = list(self.GOLD_cell_wf[cell])
            for i1 in range(len(eligibleWFs)):
                wf1 = eligibleWFs[i1]
                if wf1 not in self.GOLD_cellMates:
                    self.GOLD_cellMates[wf1] = {}
                for i2 in range(i1+1, len(eligibleWFs)):
                    wf2 = eligibleWFs[i2]
                    if wf2 not in self.GOLD_cellMates:
                        self.GOLD_cellMates[wf2] = {}

                    self.GOLD_cellMates[wf1][wf2] = True
                    self.GOLD_cellMates[wf2][wf1] = True


    def eval(self, debug=False):

        print('EVALUATING... (THIS MAY TAKE A HOT SEC)')
        total = len(self.GOLD_cellMates)
        increment = 100/total

        # take macro F score of cellMates over all forms in gold vocabulary
        Fs = []
        minF = 2
        maxF = -1
        counter = 0
        nextTen = 10
        for wf in self.GOLD_cellMates:
     
            try:
                correctSet = set(self.GOLD_cellMates[wf]).intersection(set(self.cellMates[wf]))
                correct = len(correctSet)
            except IndexError:
                correctSet = set()
                correct = 0

            if correct == 0:
                F = 0
            else:
                predicted = len(self.cellMates[wf])
                total_recall_with_Arnold_Schwarzeneggar = len(self.GOLD_cellMates[wf])
                
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
                for par in self.wf_lem[wf]:
                    print('\tLemma {}'.format(par))
                    for fts in self.GOLD_lem_wf_lstFtlists[par][wf]:
                        print('\t\t{}'.format(';'.join(fts)))
                        correctFeats[';'.join(fts)] = True

                precErrors = {}
                precErrorFts = {}
                for x in self.cellMates[wf]:
                    if x not in self.GOLD_cellMates[wf]:
                        precErrors[x] = True
                        for par in self.wf_lem[x]:
                            for fts in self.GOLD_lem_wf_lstFtlists[par][x]:
                                fts = ';'.join(fts)
                                if fts not in precErrorFts:
                                    precErrorFts[fts] = 0
                                precErrorFts[fts] += 1

                recErrors = {}
                recErrorFts = {}
                for x in self.GOLD_cellMates[wf]:
                    if x not in self.cellMates[wf]:
                        recErrors[x] = True
                        for par in self.wf_lem[x]:
                            for fts in self.GOLD_lem_wf_lstFtlists[par][x]:
                                fts = ';'.join(fts)
                                if fts not in recErrorFts:
                                    recErrorFts[fts] = 0
                                recErrorFts[fts] += 1


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


    def get_cell_distance_matrix(self):

        self.cell_cellInd = {}
        self.cellInd_cell = {}
        self.cellDistMat = np.array([[0.0]*len(self.allCells)]*len(self.allCells))

        cellList = list(self.allCells)
        for c1 in range(len(cellList)):
            cell1 = cellList[c1]
            fts1 = set(get_MSP_from_cell(self.GOLD_skeleton, cell1, option='stringList'))
            self.cell_cellInd[cell1] = c1 
            self.cellInd_cell[c1] = cell1
            for c2 in range(c1+1, len(cellList)):
                cell2 = cellList[c2]
                fts2 = set(get_MSP_from_cell(self.GOLD_skeleton, cell2, option='stringList'))

                similarity = float(len(fts1&fts2) / len(fts1|fts2))
                distance = float(1 - similarity)

                self.cellDistMat[c1][c2] = distance
                self.cellDistMat[c2][c1] = distance


    def get_medoids(self, distance_function, debug=False, embeddings=None, D1=None, D2=None, alpha=None, bias=1):  # TO ADD: CHANGE K-MEDOID CLUSTERING OUTPUT TO A SOFTMAX DISTRIBUTION TO FACILITATE DOWNSTREAM CELL REASSIGNMENTS

        # get distance metrics
        self.distMat, embeddings = self.get_distance_matrix(distance_function, D1=D1, D2=D2, embeddings=embeddings, alpha=alpha, bias=bias)

        # k-medoid clustering of distance metrics
        print('K-medoid clustering {} distance matrix'.format(distance_function))
        self.medoid_centroidInd, self.medoid_ind = kMedoids(self.distMat, len(self.allCells))
        # self.medoid_ind is a dictionary from cluster labels to an array of member ind's
        # M is a list of medoid-center ind's with indeces corresponding to cluster labels

        self.ind_centroidInd = {}
        self.ind_medoid = {}
        self.wf_medoid = {}
        self.medoid_wf = {}
        for medoid in self.medoid_ind:
            self.medoid_wf[medoid] = {}
            for ind in self.medoid_ind[medoid]:
                wf = self.ind_wf[ind]
                self.ind_medoid[ind] = medoid
                centroidInd = self.medoid_centroidInd[medoid]
                self.ind_centroidInd[ind] = centroidInd
                self.ind_medoid[ind] = medoid
                if wf not in self.wf_medoid:
                    self.wf_medoid[wf] = {}
                self.wf_medoid[wf][medoid] = True
                self.medoid_wf[medoid][wf] = True

        # MUSE(srcMat, srcMatVec, tgtMat, tgtMatVec, mapping_dir, srcDest, tgtDest, gpu=False)


    def get_clusters(self, distance_function, debug=False, embeddings=None, D1=None, D2=None, alpha=None, bias=1):

        # Get distance matrix
        self.distMat, embeddings = self.get_distance_matrix(distance_function, D1=D1, D2=D2, embeddings=embeddings, alpha=alpha, bias=bias)

        ### Get positional matrix
        posMatDat = '{}.{}_posMatrix.dat'.format(UNIMORPH_LG, EXP_ID)
        ### Check if the positional embeddings have already been computed
        try:
            self.posMat = np.load(posMatDat)
        ### If not, compute them
        except FileNotFoundError:
            self.posMat = embed_distMat(self.distMat, len(self.allCells))
            with open(posMatDat, 'wb') as outfile:
                np.save(outfile, self.posMat)

        # k-medoid clustering of distance metrics
        print('K-means clustering {} distance matrix'.format(distance_function))

        kmeans = KMeans(n_clusters=len(self.allCells)).fit(self.posMat)
        ind_cluster = kmeans.labels_
        cluster_centroid = kmeans.cluster_centers_

        self.ind_cluster = {}
        self.cluster_ind = {}
        self.wf_cluster = {}
        self.cluster_wf = {}
        for ind in range(len(ind_cluster)):
            cluster = ind_cluster[ind]
            wf = self.ind_wf[ind]
            self.ind_cluster[ind] = cluster
            if cluster not in self.cluster_ind:
                self.cluster_ind[cluster] = {}
                self.cluster_wf[cluster] = {}
            self.cluster_ind[cluster][ind] = True
            self.cluster_wf[cluster][wf] = True
            if wf not in self.wf_cluster:
                self.wf_cluster[wf] = {}
            self.wf_cluster[wf][cluster] = True

        if debug:
            for cluster in self.cluster_ind:
                print('\nCLUSTER {}'.format(cluster))
                mostCommonCells = {}
                printLines = []
                for ind in self.cluster_ind[cluster]:
                    wf = self.ind_wf[ind]
                    lem = self.ind_lem[ind]
                    x = len(self.GOLD_wf_cell[wf])
                    for ftList in self.GOLD_lem_wf_lstFtlists[lem][wf]:
                        cell = '_'.join(ftList)
                        x -= 1
                        if cell not in mostCommonCells:
                            mostCommonCells[cell] = 0
                        mostCommonCells[cell] += 1
                        if x > 0:
                            printLines.append('\t\t{}'.format(cell))
                        else:
                            printLines.append('\t\t{}\t{}  <-  {}'.format(cell, wf, lem))
                mostCommonCells = list(zip(mostCommonCells.values(), mostCommonCells.keys()))
                mostCommonCells.sort(reverse=True)
                print('\tMost frequent gold cells')
                print('\n'.join('\t\t{}\t{}'.format(q[0], q[1]) for q in mostCommonCells))
                print('\tAll cluster members with corresponding lemma and gold cells')
                for pl in printLines:
                    print(pl)


        assert len(self.ind_cluster) == len(self.ind_wf)

        # MUSE(srcMat, srcMatVec, tgtMat, tgtMatVec, mapping_dir, srcDest, tgtDest, gpu=False)


    def assignMedoid_random(self, mORc='medoid'):

        if mORc == 'medoid':
            group_ind = self.medoid_ind
        elif mORc == 'cluster':
            group_ind = self.cluster_ind
        else:
            print('UNSUPPORTED GROUPING: {}'.format(mORc))
            exit()

        self.cell_ind = {}
        self.ind_cell = {}
        self.wf_cell = {}
        self.cell_wf = {}
        self.medoid_cell = {}
        self.cell_medoid = {}

        random_cells = list(self.allCells)
        random.shuffle(random_cells)

        for medoid in group_ind:
            cell = random_cells.pop(0)
            self.cell_wf[cell] = {}
            for ind in group_ind[medoid]:
                wf = self.ind_wf[ind]
                if wf not in self.wf_cell:
                    self.wf_cell[wf] = {}
                self.wf_cell[wf][cell] = True 
                self.cell_wf[cell][wf] = True
                self.medoid_cell[medoid] = cell
                self.cell_medoid[cell] = medoid
                self.ind_cell[ind] = cell
                if cell not in self.cell_ind:
                    self.cell_ind[cell] = {}
                self.cell_ind[cell][ind] = True

        assert len(random_cells) == 0


    def assignMedoid_oracle(self, mORc='medoid'):

        if mORc == 'medoid':
            group_ind = self.medoid_ind
        elif mORc == 'cluster':
            group_ind = self.cluster_ind
        else:
            print('UNSUPPORTED GROUPING: {}'.format(mORc))
            exit()

        medoid_cell_F = []

        for medoid in group_ind:
            for cell in self.allCells:

                predicted = {}
                correct = {}
                total = dict(self.GOLD_cell_wf[cell])

                for ind in group_ind[medoid]:
                    wf = self.ind_wf[ind]
                    predicted[wf] = True
                    for lem in self.wf_lem[wf]:
                        if wf in total:
                            correct[wf] = True
                            break

                correct = len(correct)
                predicted = len(predicted)
                total = len(total)

                if correct == 0:
                    F = 0

                else:

                    prec = correct / predicted
                    rec = correct / total 
                    F = 100 * ( (2 * prec * rec) / (prec + rec) )

                medoid_cell_F.append([F, [medoid,cell]])

        medoid_cell_F.sort(reverse=True)

        available_Ms = list(group_ind)
        available_Cs = list(self.allCells)

        cell_wf = {}

        for item in medoid_cell_F:
            medoid = item[1][0]
            cell = item[1][1]
            if medoid in available_Ms and cell in available_Cs:
                available_Ms.remove(medoid)
                available_Cs.remove(cell)
                cell_wf[cell] = {}
                for ind in group_ind[medoid]:
                    cell_wf[cell][self.ind_wf[ind]] = True
                if len(cell_wf) == len(self.GOLD_cell_wf):
                    break

        oracular_assignment_F = evalFull(self.GOLD_cell_wf, cell_wf)

        return oracular_assignment_F


    def assignCells_clusterMapHybrid(self, distance_function, D1=None, D2=None, embeddings=None,debugCells = False, debugClusters=False, alpha=None, bias=1, gpu=False, topn=5):

        self.DMmap(distance_function, D1=D1, D2=D2, embeddings=embeddings, debug=debugCells, alpha=alpha, bias=bias, gpu=gpu)
        self.DMmap_assign_probabilistic(topn=topn)
        self.get_clusters(args.distance_function, debug=debugClusters, embeddings=args.embeddings, D1=args.d1, D2=args.d2, alpha=args.alpha, bias=args.bias_against_overabundance)
        
        # Get bidirectional cellMates dictionary
        self.cellMates = {}

        progress = 0
        increment = 100/len(self.ind_wf)
        nextTen = 10
        origNextTen = nextTen
        print('Getting cell mates based on cluster membership and cell assignment')
        eligibleInds = dict(self.ind_wf)
        for ind1 in self.ind_wf:
            del eligibleInds[ind1]
            progress += increment
            if progress >= nextTen:
                print('{}% complete'.format(str(round(progress, 2))))
                nextTen += origNextTen
            wf1 = self.ind_wf[ind1]
            if wf1 not in self.cellMates:
                self.cellMates[wf1] = {}
            cluster1 = self.ind_cluster[ind1]
            cell1 = self.ind_cell[ind1]

            for ind2 in eligibleInds:
                wf2 = self.ind_wf[ind2]
                cluster2 = self.ind_cluster[ind2]
                cell2 = self.ind_cell[ind2]

                if cluster1 == cluster2 or (cell1 in self.ind_bestCells[ind2] and cell2 in self.ind_bestCells[ind1]):
                    if wf2 not in self.cellMates:
                        self.cellMates[wf2] = {}
                    self.cellMates[wf1][wf2] = True 
                    self.cellMates[wf2][wf1] = True


    def DMmap_assign_probabilistic(self, prohibit_over_abundance=False, topn=5):

        # 5) assign inds and wfs to cells based on NN function, with constraints against:
            # over abundance
            # infeasible paradigm shapes according to self.metaParadigms?
        self.cell_wf = {}
        self.wf_cell = {}
        self.ind_cell = {}
        self.cell_ind = {}
        self.lem_cell_ind = {} # watch out for over abundance
        self.ind_bestCells = {}
        averted_over_abundance = 0
        for cell in self.allCells:
            self.cell_wf[cell] = {}
            self.cell_ind[cell] = {}
        for ind in range(len(self.posMat)):
            wf = self.ind_wf[ind]
            lem = self.ind_lem[ind]
            if lem not in self.lem_cell_ind:
                self.lem_cell_ind[lem] = {}
            posVec = get_vector(str(ind), self.posMat_mapped)
            n = topn
            bestCells = self.cellPosMat_mapped.wv.similar_by_vector(posVec, topn=topn)
            bestCell = self.cellInd_cell[int(bestCells[0][0])]
            self.ind_bestCells[ind] = {self.cellInd_cell[int(x[0])]:True for x in bestCells}

            # prohibit orthographic over abundance.. revisit later for unstandardized spelling
            if prohibit_over_abundance:
                if bestCell in self.lem_cell_ind[lem]:
                    averted_over_abundance += 1
                while bestCell in self.lem_cell_ind[lem]:
                    try:
                        bestCells.pop(0)
                        bestCell = self.cellInd_cell[int(bestCells[0][0])]
                    except IndexError:
                        print('Over abundance prohibited us from finding a cell match within the first {} attempted form -> cell mappings'.format(str(n)))
                        n += topn
                        bestCells = self.cellPosMat_mapped.wv.similar_by_vector(posVec, topn=n)[n-topn:]
                        bestCell = self.cellInd_cell[int(bestCells[0][0])]


            # Having identified the best cell, make the assignments
            self.cell_wf[bestCell][wf] = True
            self.cell_ind[bestCell][ind] = True
            self.ind_cell[ind] = bestCell
            if wf not in self.wf_cell:
                self.wf_cell[wf] = {}
            self.wf_cell[wf][bestCell] = True
            if bestCell not in self.lem_cell_ind:
                self.lem_cell_ind[lem][bestCell] = {}
                self.lem_cell_ind[lem][bestCell][ind] = True

        print('Changed best cell {} times out of {} to avoid over abundance'.format(str(averted_over_abundance), str(len(self.ind_cell))))


    def assignCells_DMmap(self, distance_function, D1=None, D2=None, embeddings=None, debug=False, alpha=None, bias=1, gpu=False):

        self.DMmap(distance_function, D1=D1, D2=D2, embeddings=embeddings, debug=debug, alpha=alpha, bias=bias, gpu=gpu)
        self.DMmap_assign(self)
        self.get_cellMates()


    def DMmap(self, distance_function, D1=None, D2=None, embeddings=None, debug=False, alpha=None, bias=1, gpu=False):

        # get distance metrics
        self.distMat, embeddings = self.get_distance_matrix(distance_function, D1=D1, D2=D2, embeddings=embeddings, alpha=alpha, bias=bias)

        # if bias == 1:
        #     EXP_ID = '{}_{}-{}-{}'.format(distance_function, str(D1), str(D2), str(alpha))
        # else:
        #     EXP_ID = '{}_{}-{}-{}-bias{}'.format(distance_function, str(D1), str(D2), str(alpha), str(bias))

        # 1) self.cellDistMat = cXc matrix of gold skeleton cell feature dissimilarities
            # where c is the number of cells
        self.get_cell_distance_matrix()

        # 2) self.cellPosMat = embed_distMat(self.cellDistMat, c)
            # cXc positional matrix derived from the distance matrix self.cellDistMat
        self.cellPosMat = embed_distMat(self.cellDistMat, len(self.cellDistMat))

        # 3) self.posMat = embed_distMat(self.distMat, c)
            # nXc positional matrix where n is the vocabulary size
        posMatDat = '{}.{}_posMatrix.dat'.format(UNIMORPH_LG, EXP_ID)
        ### Check if the positional embeddings have already been computed
        try:
            self.posMat = np.load(posMatDat)
        ### If not, compute them
        except FileNotFoundError:
            self.posMat = embed_distMat(self.distMat, len(self.cellDistMat))
            with open(posMatDat, 'wb') as outfile:
                np.save(outfile, self.posMat)

        # 4) self.posMat_cellMapped = MUSE(self.posMat, self.cellPosMat)
            # self.posMat mapped into self.cellPosMat's embedding space via linear transformation
        ### Get locations of relevant matrices to perform mapping
        srcMatVec = '{}.{}_posMatrix.vec'.format(UNIMORPH_LG, EXP_ID)
        tgtMatVec = '{}.{}_cellPosMatrix.vec'.format(UNIMORPH_LG, EXP_ID)
        ### Get relevant output locations
        # os.system('mkdir -p {}/{}/{}'.format(MAPPING_DIR, LG, EXP_ID))
        posMat_mapped = os.path.join(MAPPING_DIR, LG, EXP_ID, 'posMat_mapped.bin')
        cellPosMat_mapped = os.path.join(MAPPING_DIR, LG, EXP_ID, 'cellPosMat_mapped.bin')
        bestMap = os.path.join(MAPPING_DIR, LG, EXP_ID, 'map.pth')
        ### Check if the outputs have already been precomputed
        try:
            self.posMat_mapped = loadVectors(posMat_mapped, model='w2v', binary=True)
            self.cellPosMat_mapped = loadVectors(cellPosMat_mapped, model='w2v', binary=True)
            self.bestMap = bestMap
        ### If not, perform the relevant embedding mapping
        except (FileNotFoundError, NotADirectoryError):
            # Run MUSE to map embedding spaces
            self.posMat_mapped, self.cellPosMat_mapped, self.bestMap = MUSE(self.posMat, srcMatVec, self.cellPosMat, tgtMatVec, [MAPPING_DIR, LG, EXP_ID], posMat_mapped, cellPosMat_mapped, gpu=gpu)


    def DMmap_assign(self, prohibit_over_abundance=False):

        # 5) assign inds and wfs to cells based on NN function, with constraints against:
            # over abundance
            # infeasible paradigm shapes according to self.metaParadigms?
        self.cell_wf = {}
        self.wf_cell = {}
        self.ind_cell = {}
        self.cell_ind = {}
        self.lem_cell_ind = {} # watch out for over abundance
        averted_over_abundance = 0
        for cell in self.allCells:
            self.cell_wf[cell] = {}
            self.cell_ind[cell] = {}
        for ind in range(len(self.posMat)):
            wf = self.ind_wf[ind]
            lem = self.ind_lem[ind]
            if lem not in self.lem_cell_ind:
                self.lem_cell_ind[lem] = {}
            posVec = get_vector(str(ind), self.posMat_mapped)
            origN = 5
            n = origN
            bestCells = self.cellPosMat_mapped.wv.similar_by_vector(posVec, topn=n)
            bestCell = self.cellInd_cell[int(bestCells[0][0])]


            # prohibit orthographic over abundance.. revisit later for unstandardized spelling
            if prohibit_over_abundance:
                if bestCell in self.lem_cell_ind[lem]:
                    averted_over_abundance += 1
                while bestCell in self.lem_cell_ind[lem]:
                    try:
                        bestCells.pop(0)
                        bestCell = self.cellInd_cell[int(bestCells[0][0])]
                    except IndexError:
                        print('Over abundance prohibited us from finding a cell match within the first {} attempted form -> cell mappings'.format(str(n)))
                        n += origN
                        bestCells = self.cellPosMat_mapped.wv.similar_by_vector(posVec, topn=n)[n-origN:]
                        bestCell = self.cellInd_cell[int(bestCells[0][0])]


            # Having identified the best cell, make the assignments
            self.cell_wf[bestCell][wf] = True
            self.cell_ind[bestCell][ind] = True
            self.ind_cell[ind] = bestCell
            if wf not in self.wf_cell:
                self.wf_cell[wf] = {}
            self.wf_cell[wf][bestCell] = True
            if bestCell not in self.lem_cell_ind:
                self.lem_cell_ind[lem][bestCell] = {}
                self.lem_cell_ind[lem][bestCell][ind] = True

        print('Changed best cell {} times out of {} to avoid over abundance'.format(str(averted_over_abundance), str(len(self.ind_cell))))


    def get_lemPosMat(self, embeddings):

        ### get average embedding for each paradigm
        self.lemPosMat = []
        self.lem_lemInd = {}
        self.lemInd_lem = {}

        eligibleLems = list(self.lem_wf)
        eligibleLems.sort()
        l = len(eligibleLems)
        dim = dimensionality(embeddings)

        lemInd = -1
        for lem in eligibleLems:
            lemInd += 1
            self.lem_lemInd[lem] = lemInd
            self.lemInd_lem[lemInd] = lem
            self.lemPosMat.append(stat_tracker(type='array', len=dim))
            for wf in self.lem_wf[lem]:
                self.lemPosMat[-1].add_instance(get_vector(wf, embeddings))
        for q in range(len(self.lemPosMat)):
            self.lemPosMat[q].normalize()
            self.lemPosMat[q] = self.lemPosMat[q].mean

        ### get distances between paradigms' average embeddigns
        # self.lemDistMat = 1-pairwise_distances(self.lemPosMat, metric="cosine")


    def get_distance_matrix(self, distance_function, embeddings=None, D1= None, D2=None, alpha=None, bias=1):

        ### Make sure distance function is supported
        if distance_function not in ['lev', 'weightedLev', 'cosine', 'cosine_par', 'interpolated']:
            print('UNSUPPORTED DISTANCE METRIC: {}'.format(distance_function))
            exit()

        ### Get distance matrix filename
        if distance_function == 'interpolated':
            if bias == 1:
                filename = '{}.{}_{}-{}-{}_distMatrix.dat'.format(UNIMORPH_LG, distance_function, D1, D2, str(alpha))
            else:
                filename = '{}.{}_{}-{}-{}-bias{}_distMatrix.dat'.format(UNIMORPH_LG, distance_function, D1, D2, str(alpha), str(bias))
        else:
            if distance_function == 'weightedLev' and bias != 1:
                filename = '{}.{}-bias{}_distMatrix.dat'.format(UNIMORPH_LG, distance_function, str(bias))
            else:
                filename = '{}.{}_distMatrix.dat'.format(UNIMORPH_LG, distance_function)

        ### Check if distance matrix has been precomputed
        n = len(self.ind_wf)
        try:
            print('Checking for cached {}x{} {} distance matrix'.format(str(n), str(n), distance_function))
            # matrix1 = np.memmap(filename, dtype='float64', mode='r', shape=(n, n))
            matrix = np.load(filename)

        ### Begin building distance matrix
        except FileNotFoundError:

            print('None found.. Building {}x{} {} distance matrix'.format(str(n), str(n), distance_function))
            matrix = np.array([[0.0]*n]*n)

            ### Gather prerequisite data to inform distance matrix
            if distance_function == 'interpolated':
                ### composite distance matrices to be interpolated
                D1, embeddings = self.get_distance_matrix(D1, embeddings=embeddings, bias=bias)
                D2, embeddings = self.get_distance_matrix(D2, embeddings=embeddings, bias=bias)
                ### load word embeddings
            elif distance_function in ['cosine', 'cosine_par']:
                print('Loading word embeddings from {}'.format(embeddings))
                embeddings = loadVectors(embeddings)
                ### get paradigm--paradigm distance matrix
                if distance_function == 'cosine_par':
                    self.get_lemPosMat(embeddings)
                    self.p2_wf1_p1 = {}
                    self.wf1_p1 = {}

            ### Track progress
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
                    wf1 = self.ind_wf[i]
                    wf2 = self.ind_wf[j]
                    p1 = self.ind_lem[i]
                    p2 = self.ind_lem[j]

                    if p1 == p2:
                        distance = 1

                    else:

                        ### get pairwise distance
                        if distance_function == 'weightedLev':
                            distance = 1 - self.get_likelihood_cellMate_weighted(p1, wf1, p2, wf2)
                        elif distance_function == 'lev':
                            distance = 1 - self.get_likelihood_cellMate_unweighted(p1, wf1, p2, wf2)
                        elif distance_function == 'cosine':
                            distance = cosDist(wf1, wf2, embeddings)
                        elif distance_function == 'cosine_par':
                            distance = self.get_parConditional_cosDist(wf1, wf2, p1, p2, embeddings)
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


    def get_parConditional_cosDist(self, wf1, wf2, p1, p2, embeddings):

        try:
            analVec = self.p2_wf1_p1[p2][wf1][p1]
        except KeyError:
            try:
                B = self.wf1_p1[wf1][p1]
                l2 = self.lem_lemInd[p2]
                l2vec = self.lemPosMat[l2]
                analVec = l2vec + B

                if p2 not in self.p2_wf1_p1:
                    self.p2_wf1_p1[p2] = {}
                if wf1 not in self.p2_wf1_p1[p2]:
                    self.p2_wf1_p1[p2][wf1] = {}
                self.p2_wf1_p1[p2][wf1][p1] = analVec 

            except KeyError:
                l1 = self.lem_lemInd[p1]
                l2 = self.lem_lemInd[p2]
                l1vec = self.lemPosMat[l1]
                l2vec = self.lemPosMat[l2]
                wf1vec = get_vector(wf1, embeddings)

                B = wf1vec - l1vec
                if wf1 not in self.wf1_p1:
                    self.wf1_p1[wf1] = {}
                self.wf1_p1[wf1][p1] = B 

                analVec = l2vec + B

                if p2 not in self.p2_wf1_p1:
                    self.p2_wf1_p1[p2] = {}
                if wf1 not in self.p2_wf1_p1[p2]:
                    self.p2_wf1_p1[p2][wf1] = {}
                self.p2_wf1_p1[p2][wf1][p1] = analVec  

        wf2vec = get_vector(wf2, embeddings)
        dist = 1 - ( (cosine_similarity([analVec],[wf2vec])[0][0] + 1) / 2 )
        
        return dist


    def get_likelihood_cellMate_weighted(self, p1, wf1, p2, wf2):

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


    def get_likelihood_cellMate_unweighted(self, p1, wf1, p2, wf2):

        numerator = 0
        denominator = len(wf1) + len(wf2)
        blocks = lv.matching_blocks(lv.editops(''.join(wf1), ''.join(wf2)), len(wf1), len(wf2))
        for match in blocks:
            for i in range(match[0], match[0]+match[2]):
                numerator += 2

        return numerator/denominator


    def evalFull(self, debug=False):

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

            if debug:

                print('\n\n{}\t{} F-score'.format(get_MSP_from_cell(self.GOLD_skeleton, cell), str(round(F, 2))))

                precErrors = {}
                precErrorFts = {}
                precErrors = predicted - correct
                for wf in precErrors:
                    for lem in self.wf_lem[wf]:
                        for fts in ana.GOLD_lem_wf_lstFtlists[lem][wf]:
                            fts = ';'.join(fts)
                            if fts not in precErrorFts:
                                precErrorFts[fts] = 0
                            precErrorFts[fts] += 1

                recErrors = {}
                recErrorFts = {}
                recErrors = actual - correct
                for wf in recErrors:
                    for lem in self.wf_lem[wf]:
                        for fts in ana.GOLD_lem_wf_lstFtlists[lem][wf]:
                            fts = ';'.join(fts)
                            if fts not in recErrorFts:
                                recErrorFts[fts] = 0
                            recErrorFts[fts] += 1

                rankedPrecErrorFts = list(zip(list(precErrorFts.values()), list(precErrorFts.keys())))
                rankedPrecErrorFts.sort(reverse=True)
                rankedPrecErrorFts = '  '.join('{}_({})'.format(l[1], str(l[0])) for l in rankedPrecErrorFts)

                rankedRecErrorFts = list(zip(list(recErrorFts.values()), list(recErrorFts.keys())))
                rankedRecErrorFts.sort(reverse=True)
                rankedRecErrorFts = '  '.join('{}_({})'.format(l[1], str(l[0])) for l in rankedRecErrorFts)

                print('\tCORRECT:\n{}'.format(', '.join(list(correct))))
                print('\tPREC ERRORS:\n{}\n{}'.format(rankedPrecErrorFts, ', '.join(list(precErrors))))
                print('\tREC ERRORS:\n{}\n{}'.format(rankedRecErrorFts, ', '.join(list(recErrors))))

        macroF = numer / denom
        strMacroF = str(round(macroF, 3))
        strMinF = str(round(minF, 3))
        strMaxF = str(round(maxF, 3))

        print('\n\nMacro F: {}\t(min: {}, max: {}, total cells: {})'.format(strMacroF, strMinF, strMaxF, str(len(self.cell_wf))))

        return macroF


    def get_ind_silhoette(self):
        print('Getting index silhoettes')
        label_vector = np.array([0]*len(self.distMat))
        for label in self.cell_ind:
            for ind in self.cell_ind[label]:
                label_vector[ind] = label
        self.ind_silhoette = silhouette_samples(self.distMat, label_vector, metric='precomputed')

        self.least_cohesive_inds = []
        for ind in range(len(self.ind_silhoette)):
            self.least_cohesive_inds.append([self.ind_silhoette[ind], ind])
        self.least_cohesive_inds.sort()


    def get_cell_cohesion(self):
        print('Calculating cell cohesion')
        self.cell_cohesion = {}  # average of composite silhoettes
        for ind in range(len(self.ind_silhoette)):
            cell = self.ind_cell[ind]
            if cell not in self.cell_cohesion:
                self.cell_cohesion[cell] = stat_tracker()
                self.cell_cohesion[cell].add_instance(self.ind_silhoette[ind])
        for cell in self.cell_cohesion:
            self.cell_cohesion[cell].normalize()

        self.least_cohesive_cells = []
        for cell in self.cell_cohesion:
            self.least_cohesive_cells.append([self.cell_cohesion[cell].cohesion, cell])
        self.least_cohesive_cells.sort()


###################################################################################
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def embed_distMat(dist_matrix, dim):
    mds = manifold.MDS(n_components=dim, dissimilarity="precomputed")
    results = mds.fit(dist_matrix)

    posMat = results.embedding_

    return posMat
    
def interpolate(v1, v2, alpha):
    value = alpha*v1 + (1-alpha)*v2
    return value

def print_sample_paradigms(ana, limit, output):

    output = open(output, 'w')

    randLems = list(ana.GOLD_lem_wf_lstFtlists)
    random.shuffle(randLems)

    try:
        limit = int(limit)

        for lem in randLems:
            for wf in ana.GOLD_lem_wf_lstFtlists[lem]:
                for fts in ana.GOLD_lem_wf_lstFtlists[lem][wf]:
                    output.write('{}\t{}\t{}\n'.format(lem, wf, ';'.join(fts)))
                    limit -= 1
            if limit < 1:
                break

    except:

        for lem in randLems:
            for wf in ana.GOLD_lem_wf_lstFtlists[lem]:
                for fts in ana.GOLD_lem_wf_lstFtlists[lem][wf]:
                    if limit in fts:
                        output.write('{}\t{}\t{}\n'.format(lem, wf, ';'.join(fts)))

    output.close()

    exit()

def MUSE(srcMat, srcMatVec, tgtMat, tgtMatVec, mapping_dir, srcDest, tgtDest, gpu=False):

    ### Format vocabulary and cell, i.e., source and tgt matrices for MUSE
    print('Formatting source embeddings for MUSE')
    emb_dim = print_matrix_as_FTvec(srcMat, srcMatVec)
    print('Formatting target embeddings for MUSE')
    assert emb_dim == print_matrix_as_FTvec(tgtMat, tgtMatVec)

    ### Learn linear transformation from source to target embedding space
    print('MUSE is learning a linear transformation from {} to {} embedding space'.format(srcMatVec.split('/')[-1].strip('.vec'), tgtMatVec.split('/')[-1].strip('.vec')))
    # if gpu:
    #     command = 'python {} --src_lang src --tgt_lang tgt --src_emb {} --tgt_emb {} --cuda {}, --emb_dim {} --exp_path {} --exp_name {} --exp_id {}'.format(os.path.join(ANA_DIR, 'MUSE/unsupervised_ANA.py'), srcMatVec, tgtMatVec, str(gpu), emb_dim, mapping_dir[0], mapping_dir[1], mapping_dir[2])
    # else:
    #     
    command = 'python {} --src_lang src --tgt_lang tgt --src_emb {} --tgt_emb {} --emb_dim {} --exp_path {} --exp_name {} --exp_id {}'.format(os.path.join(ANA_DIR, 'MUSE/unsupervised_ANA.py'), srcMatVec, tgtMatVec, emb_dim, mapping_dir[0], mapping_dir[1], mapping_dir[2])
    os.system(command)

    ### Gather MUSE output
    mapping_dir = os.path.join(mapping_dir[0], mapping_dir[1], mapping_dir[2])
    best_mapping = os.path.join(mapping_dir, 'best_mapping.pth')
    src_emb = os.path.join(mapping_dir, 'vectors-src.txt')
    tgt_emb = os.path.join(mapping_dir, 'vectors-tgt.txt')
    ### Load mapped embeddings
    src_emb = loadVectors(src_emb, model='w2v', binary=False)
    tgt_emb = loadVectors(tgt_emb, model='w2v', binary=False)
    ### Save mapped embeddings
    saveVectors(src_emb, srcDest, model='w2v', binary=True)
    saveVectors(tgt_emb, tgtDest, model='w2v', binary=True)

    return src_emb, tgt_emb, best_mapping

def print_matrix_as_FTvec(matrix, matrixVec):

    matrixVec = open(matrixVec, 'w')
    emb_dim = str(len(matrix[0]))
    matrixVec.write('{} {}\n'.format(str(len(matrix)), emb_dim))
    for ind in range(len(matrix)):
        matrixVec.write('{} {}\n'.format(str(ind), ' '.join(str(round(x, 5)) for x in matrix[ind])))
    matrixVec.close()

    return emb_dim

class stat_tracker:

    def __init__(self, type=float, len=0):

        self.size = 0
        self.type = type
        if self.type == float:
            self.sum = 0.0
        elif self.type == 'array':
            self.sum = np.array([0.0]*len)

    def add_instance(self, value):

        if self.type == 'array':
            value = np.array(value)
        self.size += 1
        self.sum += value

    def mult_instance(self, instances, value):

        if self.type == 'array':
            value = np.array(value)
        self.size += instances
        self.sum += value*instances

    def div_instance(self, instances, value):

        if self.type == 'array':
            value = np.array(value)
        self.size -= instances
        self.sum -= value*instances

    def sub_instance(self, value):

        if self.type == 'array':
            value = np.array(value)
        self.size -= 1
        self.sum -= value

    def normalize(self):

        self.mean = self.sum / self.size 

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

def get_MSP_from_cell(skeleton, cell, option='string'):

        ftList = []
        coord = list(int(x) for x in cell.split(','))

        for dimInd in range(len(coord)):
            featInd = coord[dimInd]
            if featInd != 0:

                if option in ['string', 'stringList']:
                    ftList.append(skeleton[dimInd][featInd])
                elif option == 'list':
                    ftList.append([featInd])
                else:
                    print('DID NOT RECOGNIZE MSP REPRESENTATION FORMAT {}'.format(option))
                    exit()
        if option == 'string':
            ftList = ';'.join(ftList)

        return ftList

def evalFull(gold_cell_wf, cell_wf):

        numer = 0
        denom = 0
        minF = 101
        maxF = -1

        for cell in gold_cell_wf:

            predicted = set(cell_wf[cell])
            actual = set(gold_cell_wf[cell])
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

        print('\n\nMacro F: {}\t(min: {}, max: {}, total cells: {})'.format(strMacroF, strMinF, strMaxF, str(len(cell_wf))))

        return macroF


###################################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--unimorph', type=str, help='Location of Unimorph file', required=True)
    parser.add_argument('-e', '--embeddings', type=str, help='Location of word embeddings', required=False, default=None)
    parser.add_argument('-l', '--limit', type=str, help='Shall we draw a sample of the full unimorph data? You can limit it by number of instances or by shared feature.', required=False, default=None)
    parser.add_argument('-d', '--distance_function', type=str, choices=['interpolated', 'weightedLev', 'lev', 'cosine', 'cosine_par'], help='Metric that determines the relevant distance matrix used for cell clustering', required=False, default='interpolated')
    parser.add_argument('-x', '--d1', type=str, choices=['weightedLev', 'lev', 'cosine', 'cosine_par'], help='First metric considered in the interpolated distance function', required=False, default='weightedLev')
    parser.add_argument('-y', '--d2', type=str, choices=['weightedLev', 'lev', 'cosine', 'cosine_par'], help='Second metric considered in the interpolated distance function', required=False, default='cosine')
    parser.add_argument('-a', '--alpha', type=float, choices=[continuous_range(0.0, 1.0)], help='Relative weight placed on d1 for the interpolated distance metric', required=False, default=0.5)
    parser.add_argument('-n', '--topn', type=int, help='Top n nearest neighbors to look for cell mates in', required=False, default=1)
    parser.add_argument('-b', '--bias_against_overabundance', type=float, help='Bias against overabundance', required=False, default=1)
    parser.add_argument('-m', '--debugCellMates', type=str2bool, help='Show cellMates by word form during evaluation for debugging/development purposes', required=False, default=False)
    parser.add_argument('-C', '--debugClusters', type=str2bool, help='Show gold features present in each cluster during kmeans clustering for debugging/development purposes', required=False, default=False)
    parser.add_argument('-c', '--debugCells', type=str2bool, help='Show gold features present in each cell during assignment for debugging/development purposes', required=False, default=False)
    parser.add_argument('-g', '--gpu', type=str2bool, help='Show gold features present in each cell during assignment for debugging/development purposes', required=False, default=True)
    parser.add_argument('-s', '--strategy', type=str, help='How to choose cell mates and/or perform cell assignment', required=False, choices=['random', 'kmeans', 'kmedoids', 'map_voc-cell', 'kmeans_map'], default='kmeans')

    args = parser.parse_args()


    UNIMORPH_LG = args.unimorph
    LG = os.path.basename(UNIMORPH_LG)
    if args.bias_against_overabundance == 1:
        EXP_ID = '{}_{}-{}-{}'.format(args.distance_function, str(args.d1), str(args.d2), str(args.alpha))
    else:
        EXP_ID = '{}_{}-{}-{}-bias{}'.format(args.distance_function, str(args.d1), str(args.d2), str(args.alpha), str(args.bias_against_overabundance))

    ########################### READ IN DATA ######################################
    ### Read in file, get UG, Gold, plain (Gold without tags) data/array/skeletons
    ana = ANA(UNIMORPH_LG)
    ### Generate a sample file if requested
    if args.limit != None:
        print_sample_paradigms(ana, args.limit, '{}.sample-{}'.format(UNIMORPH_LG, args.limit))

    ## Get all possible array cells and meta paradigms according to Gold
    ana.get_attested_cells()    # in ara, 196 attested cells, 19 meta paradigms
                                    # in deu,  37 attested cells,  9 meta paradigms


    ########################### INITIALIZATION ######################################
    ### Assign cells based on these (potentially interpolated) metrics:
        # random, weighted Lev ED, Lev ED, and/or cosine similarity

    ana.learn_exponence_weights(bias_against_overabundance=args.bias_against_overabundance)

    oracle_assF = None
    ### RANDOM CELL ASSIGNMENT ###
    if args.strategy == 'random':
        ana.assignCells_random()
    #############################

    ### CLUSTERING BY KMEANS ###
    elif args.strategy == 'kmeans':
        ana.get_clusters(args.distance_function, debug=args.debugClusters, embeddings=args.embeddings, D1=args.d1, D2=args.d2, alpha=args.alpha, bias=args.bias_against_overabundance)
        ana.assignMedoid_random(mORc='cluster')
        ana.get_cellMates()
        oracle_assF = ana.assignMedoid_oracle(mORc='cluster')
    #############################

    ### CLUSTERING BY MEDOID ###
    elif args.strategy == 'kmedoids':
        ana.get_medoids(args.distance_function, debug=False, embeddings=args.embeddings, D1=args.d1, D2=args.d2, alpha=args.alpha, bias=args.bias_against_overabundance)
        ana.assignMedoid_random()
        ana.get_cellMates()
        oracle_assF = ana.assignMedoid_oracle()
    #############################

    ### MAPPING POSITIONAL VOCAB INTO POSITIONAL CELL EMBEDDINGS ###
    elif args.strategy == 'map_voc-cell':
        ana.assignCells_DMmap(args.distance_function, D1=args.d1, D2=args.d2, embeddings=args.embeddings, debug=args.debugCells, alpha=args.alpha, bias=args.bias_against_overabundance, gpu=args.gpu) # debug will show feats and wf's of proposed cells
    #############################

    ### CONSIDER BOTH KMEANS CLUSTERING AND VOCAB-TO-CELL MAP WHEN CHOOSING CELLMATES ###
    elif args.strategy == 'kmeans_map':
        ana.assignCells_clusterMapHybrid(args.distance_function, D1=args.d1, D2=args.d2, embeddings=args.embeddings, debugCells=args.debugCells, debugClusters=args.debugClusters, alpha=args.alpha, bias=args.bias_against_overabundance, gpu=args.gpu, topn=args.topn) # debug will show feats and wf's of proposed cells
    #############################



    ############################### EVALUATION ######################################
    ### Evaluate cell mates
    macroF = ana.eval(debug=args.debugCellMates) # debug shows wf's F,p,r, and cellMates
    ### Secondary evaluation on cell assignment - may not be relevant
    cell_macroF = ana.evalFull(debug=args.debugCells)


    ##################################### SUMMARY #####################################
    print('\n\nCell Mate Prediction Per Word Macro F-score: {}'.format(str(round(macroF, 2))))
    print('Cell Assignment Per Cell Macro F-score: {}'.format(str(round(cell_macroF, 2))))
    # if cluster by medoid
    if oracle_assF != None:
        print('Oracle Cell Assignment Per Cell Macro F-score {}'.format(str(round(oracle_assF, 2))))




    

