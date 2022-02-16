from scipy.stats import pearsonr
import argparse
from nltk import word_tokenize
from util import parse_sts
# can just use symmetrical_nist function from sts_nist.py
# but in order to keep the consistent code format of this assignment, an extra function is defined
# from sts_nist import symmetrical_nist
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance
import jiwer
from scipy.stats import pearsonr


def getToksFromPairTexts(text_pair):
    t1, t2 = text_pair

    # tokenize
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())

    return t1_toks, t2_toks


def getNISTScore(t1_toks, t2_toks):
    try:
        nist_1 = sentence_nist([t1_toks, ], t2_toks)
    except ZeroDivisionError:
        nist_1 = 0.0

    try:
        nist_2 = sentence_nist([t2_toks, ], t1_toks)
    except ZeroDivisionError:
        nist_2 = 0.0
    nist_pair = nist_1 + nist_2
    return nist_pair


def getBLEUScore(t1_toks, t2_toks):
    sf = SmoothingFunction()
    try:
        bleu_1 = sentence_bleu([t1_toks, ], t2_toks, smoothing_function=sf.method0)
        bleu_2 = sentence_bleu([t2_toks, ], t1_toks, smoothing_function=sf.method0)
        bleu_pair = bleu_1 + bleu_2
    except ZeroDivisionError:
        bleu_pair = 0
    return bleu_pair


def getWER(pair_text):
    t1, t2 = pair_text
    # Reference: https://pypi.org/project/jiwer/
    # get tidy texts
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
    ])
    # get word error rate
    wer_pair = jiwer.wer(
        t1,
        t2,
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    return wer_pair


# reference: https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/
def getLCS_recursive(X, Y, m, n):
    """find the length of longest subsequence present in two text sequences
    args:
        - X: a list of tokens of sequence 1
        - Y:a list of tokens of sequence 2
        - m: the length of list X
        - n:the length of list Y
    return:
        - a integer indicating the length of the longest common subsequence
    """
    if m == 0 or n == 0:
        return 0
    # If last characters of both sequences match
    elif X[m-1] == Y[n-1]:
        # add 1 to the length of subsequence and call the getLCS function recursively
        # to compare the next character in the subsequence
        return 1 + getLCS_recursive(X, Y, m-1, n-1)
    # if the last characters of two sequences don't match
    else:
        # call the getLCS function recursively on both sequences to compare the next character of them
        return max(getLCS_recursive(X, Y, m, n-1), getLCS_recursive(X, Y, m-1, n))


# the recursive LCS function takes too much time for my laptop, try the tabulated implementation for the LCS problem
def getLCS(X, Y):
    """find the length of longest subsequence present in two text sequences
    This algorithm builds L[m + 1][n + 1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    args:
        - X: a list of tokens of sequence 1
        - Y:a list of tokens of sequence 2
    return:
        - a integer indicating the length of the longest common subsequence
    """
    m = len(X)
    n = len(Y)
    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def getEditDistance(t1_toks, t2_toks):
    ed_dist = edit_distance(t1_toks, t2_toks)
    return ed_dist


def check_symmetry(x, y):
    """Use assertion to verify the metric is symmetrical
    if the assertion holds, execution continues. If it does not, the program crashes
    """
    assert x == y, f"The metric is not symmetrical! Got {x} and {y}"


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)
    print(f"Found {len(texts)} STS pairs")
    # # take a sample of sentences so the code runs fast for faster debugging
    # sample_text = texts[120:140]
    # sample_labels = labels[120:140]

    data = zip(labels, texts)
    # data = zip(sample_labels, sample_text)

    # TODO 2: Calculate each of the metrics here for each text pair in the dataset
    nist_scores = []
    bleu_scores = []
    wer_scores = []
    lcs_scores = []
    edit_dist_scores = []
    for label, pair_text in data:
        # get tokens of each sentence
        t1_toks = getToksFromPairTexts(pair_text)[0]
        t2_toks = getToksFromPairTexts(pair_text)[1]

        # Check the symmetry of metrics simultaneously
        # NIST
        nist_pair1 = getNISTScore(t1_toks, t2_toks)
        nist_pair2 = getNISTScore(t2_toks, t1_toks)
        check_symmetry(nist_pair1, nist_pair2)
        nist_scores.append(nist_pair1)
    # print(nist_scores[0:10])

        # BLEU
        bleu_pair1 = getBLEUScore(t1_toks, t2_toks)
        bleu_pair2 = getBLEUScore(t2_toks, t1_toks)
        check_symmetry(bleu_pair1, bleu_pair2)
        bleu_scores.append(bleu_pair1)
    # print(bleu_scores[0:10])

        # WER
        wer_pair1 = getWER(pair_text)
        wer_pair2 = getWER(pair_text)
        check_symmetry(wer_pair1, wer_pair2)
        wer_scores.append(wer_pair1)
    # print(wer_scores[0:10])

        # LCS
        # lcs_pair = getLCS_recursive(t1_toks, t2_toks, len(t1_toks), len(t2_toks))
        lcs_pair1 = getLCS(t1_toks, t2_toks)
        lcs_pair2 = getLCS(t1_toks, t2_toks)
        check_symmetry(lcs_pair1, lcs_pair2)
        lcs_scores.append(lcs_pair1)
    # print(lcs_scores[0:10])

        # Edit Distance
        edi_dist_pair1 = getEditDistance(t1_toks, t2_toks)
        edi_dist_pair2 = getEditDistance(t1_toks, t2_toks)
        check_symmetry(edi_dist_pair1, edi_dist_pair2)
        edit_dist_scores.append(edi_dist_pair1)
    # print(edit_dist_scores[0:10])

    # TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README

    # the result of pearsonr is a tuple
    # with the Pearson’s correlation coefficient as its first element and the p-value as its second element
    # NIST
    nist_prs = pearsonr(labels, nist_scores)[0]
    # print(nist_prs)
    # BLEU
    bleu_prs = pearsonr(labels, bleu_scores)[0]
    # print(bleu_prs)
    # WER
    wer_prs = pearsonr(labels, wer_scores)[0]
    # print(wer_prs)
    # LCS
    lcs_prs = pearsonr(labels, lcs_scores)[0]
    # print(lcs_prs)
    # Edit Distance
    edit_dist_prs = pearsonr(labels, edit_dist_scores)[0]
    # print(edit_dist_prs)

    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]
    scores = [nist_prs, bleu_prs, wer_prs, lcs_prs, edit_dist_prs]
    # get the dictionary of the pearson results
    score_dict = dict(zip(score_types, scores))
    print(f"Semantic textual similarity for {sts_data}\n")
    print(score_dict)

    # # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

