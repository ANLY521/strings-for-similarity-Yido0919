Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO: Describe each metric in ~ 1 sentence**

- BLEU: evaluate the scores which are calculated for individual sentences by comparing them with a set of good quality reference translations according to human translation/judgements.
- NIST: it's based on the BLEU metric, but NIST also calculates how informative a particular n-gram is then assign more weights to it, while BLEU simply calculates n-gram precision adding equal weight to each one. 
- LCS: establish a similarity measure by finding the length of longest subsequence present in two text sequences
- Edit Distance: quantify how dissimilar two strings are to one another by counting the minimum number of operations (insertions, deletions or substitutions) required to transform one string into the other.
- WER: it's based on the edit distance metric, but it addresses the difficulty that the recognized word sequence can have a different length from the reference word sequence by first aligning the recognized word sequence with the reference (spoken) word sequence using dynamic string alignment.

**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

Metric | Train | Dev | Test 
------ | ----- | --- | ----
NIST | 0.496 | 0.593 | 0.475
BLEU | 0.371 | 0.433 | 0.353
WER | -0.324 | -0.452| -0.326
LCS | 0.385 | 0.468| 0.344
Edit Dist | -0.019 | -0.175| -0.096

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).

python sts_pearson.py --sts_data stsbenchmark/stsbenchmark/sts-train.csv

python sts_pearson.py --sts_data stsbenchmark/stsbenchmark/sts-test.csv

## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.