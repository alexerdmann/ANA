# ANA

Unsupervised Morphological Analysis. Given unimorph-style input, ANA attempts to learn the morpho-syntactic property set (MSPS)--AKA features, AKA cell--of each instance given the other information present in the instance, i.e., the word form realizing said cell and the paradigm to which the word form belongs (we do not actually use the lemma's form provided by Unimorph, but integerize lemmata.. i.e., we assume the paradigm membership is known but not the lemma from which all members are derived). In order to learn cells given word form and paradigm membership, ANA can additionally leverage gold information regarding the possible meta-paradigm structures, i.e., the possible cells that can co-occur within a paradigm.

Because ANA takes as given the oracular knowledge of what paradigm each word belongs to, ANA is intended to leverage as input the output of an unsupervised paradigm clustering algorithm, like that of [Erdmann et al., 2018](http://www.aclweb.org/anthology/W18-5806). In other words, unlike that task of identifying paradigm mates, ANA uses the knowledge of paradigm mates to identify cell mates, and to then determine which cell is occupied by each cluster of cell mates.

ANA is currently under development.. At present, this repository only reflects initialization experiments.

## Usage

For a quick demonstration, the following will embed a sample of German Unimorph vocabulary using a Levenshtein edit distance metric weighted by paradigm conditional likelihood to realize exponents. The embeddings are then k-means clustered. Gold cells are printed during clustering for debugging/development purposes. Then finally, clusters are evaluated based on the macro-F score across all word forms of how many of said word form's ground truth cell mates share at least one cluster with the word form. I say at least one cluster, because any given word form, due to syncretism and extra-paradigm ambiguity, can recieve multiple embeddings and thus appear in multiple clusters.

```
python ANA.py -u Unimorph/deu/deu.sample-N.sample-5000 -s kmeans -d weightedLev -C True
```

### Options
* *-u* Unimorph file.
* *-s* Strategy for identifying cell mates/assign clusters to cells. Options include random clustering/assignment, kmeans clustering, kmedoids clustering directly on the distance matrix, map_voc-cell which maps the embedded vocabulary into a heuristically/scholar-seeded cell embedding space and uses nearest neighborhoods in mapped embedding space to cluster, and kmeans_map which counts any cell mate pair proposed by either the map_voc-cell or the kmeans strategy to be cell mates. The latter performs the best, though kmeans also performs well on its own and is far faster/more tractible than running MUSE, which is required by the strategies involving embedding space mapping. 
* *-d* Distance metric. Interpolated performs the best, which by default interpolates weightedLev with cosine distance. Other supported metrics include an unweighted lev and a paradigm conditional cosine_par which considers the distance between analogies computed from word embeddings and the average embedding of members their respective paradigms. In practice, this takes far longer to compute than cosine distance and marginally outperforms cosine if at all, likely due to noise in the induced average paradigm embeddings.
* *-x* First distance metric to be interpolated.
* *-y* Second distance metric to be interpolated.
* *-a* Coefficient used to weight the linear interpolation of distance metrics, 0.5 by default, i.e., equal weight.
* *-e* Word embeddings used to derive the cosine distance matrix. I assume these have been trained using Gensim's Fasttext implementation.
* *-l* Limit, either an integer or a morpho-syntactic feature, used to draw a sample of either only *l* instances from the provided unimorph file, or only instances containing the feature *l*. If *l* is an integer, it will be assumed to be a limit on the number of instances, otherwise, *l* is assumed to be a morpho-syntactic feature.
* *-g* True by defualt, this boolean specifies that GPU should be used if available.
* *-C* Displays the gold cells of cluster members as clusters are formed for debug/development purposes. The argument takes a boolean, with default, False 
* *-m* Displays the gold cells during cell mate clustering for debugging/development purposes. The argument takes a boolean, with default, False 
* *-c* Displays the gold cells for final learned clusters after cell assignment for debug/development purposes. The argument takes a boolean, with default, False 
