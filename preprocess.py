from collections import Counter


def preprocess(bigrams=False, bigram_filename='data/useful_bigrams.txt'):
    """
    :param bigrams: True or False. Should bigrams be generated or not?
    :param bigram_filename: filename containing the useful bigrams to consider
    :return: vocab, X, Y, label

    Steps:
        Split into sentences
        Split sentence into words
        Remove stopwords and lemmatize / stem words
        Generate ngrams
    """
    pass


def generate_bigrams(threshold=40, separator="_"):
    """
    Given text, generate pairs of useful bigrams

    Formula to generate bigram:
        is_valid_pair(a,b) = count(a, b) - threshold / (count(a) * count(b))

    :returns : list of useful bigrams

    >> [a_b, c_d, ]
    """
    return useful_bigrams
