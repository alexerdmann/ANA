# ANA

Unsupervised Morphological Analysis. Given unimorph-style input, ANA learns to identify the morpho-syntactic property sets (MSPS) corresponding to wordforms realizing cells within a paradigm. To do this, ANA leverages gold information regarding the possible meta-paradigm structures, i.e., the possible combinations of MSPSs that can constitute a paradigm. Additionally, ANA takes as given the oracular knowledge of what paradigm each word belongs to, as ANA is meant to leverage the output of an unsupervised paradigm clustering algorithm, like that of [Erdmann et al., 2018](http://www.aclweb.org/anthology/W18-5806).

In other words, unlike that task which identifies paradigm-mates, ANA uses the knowledge of paradigm-mates to identify cell mates, and to then determine which cell is occupied by each cluster of cell mates.

ANA is currently under development, currently, the initialization process is approaching completion. More to follow shortly..

## Usage

For a quick demonstration, the following will learn a weighted Levenshtein distance matrix over the vocabulary conditional on paradigm membership and use it to k-medoid cluster the vocabulary into the attested cells realized in 5000(ish) instance sample of German verbs drawn randomly from all verbs in the official Unimorph release. After clustering, it will evaluate 

```
python ANA.py -u Unimorph/deu/deu.sample-N.sample-5000 -d weightedLev
```

### options
* *-u* Unimorph file
* *-d* Distance metric.. interpolated performs the best, which by default interpolates weightedLev with cosine distance
* *-x* The first distance metric to be interpolated
* *-y* The second distance metric to be interpolated
* *-a* The coefficient used to weight the linear interpolation of distance metrics
* *-e* The word embeddings used to derive the cosine distance matrix.. I assume these have been trained using Gensim's Fasttext implementation
* *-l* Limit, either an integer or a morpho-syntactic feature, used to draw a sample of either only *l* instances from the provided unimorph file, or only instances containing the feature *l*.. if *l* is an integer, it will be assumed to be a limit on the number of instances, otherwise, *l* is assumed to be a morpho-syntactic feature
* *-m* Displays the medoids as they clustered for debug/development purposes.. the argument expects a boolean with default False
* *-c* Displays the final learned clusters of cell mates for debug/development purposes.. the argument expects a boolean with default False
