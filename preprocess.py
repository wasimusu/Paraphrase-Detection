from collections import Counter
import os
from bisect import bisect_left

import nltk
from nltk import WordPunctTokenizer, PunktSentenceTokenizer, PorterStemmer
from nltk.tokenize import word_tokenize

sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = WordPunctTokenizer()
stemmer = PorterStemmer()


def in_sorted_list(lists, item):
    index = bisect_left(lists, item)
    if lists[min(index, len(lists) - 1)] == item:
        return True
    else:
        return False


def is_stopwords(stopwords, word):
    """
    Is word in the list of stopwords
    stopwords should be sorted for this to work
    """
    return in_sorted_list(stopwords, word)


def is_useful_bigram(bigrams, bigram):
    return in_sorted_list(bigrams, bigram)


def generate_bigrams(text, save_filename='data/useful_bigrams.txt', read_percent=1, bigram_percent=0.05):
    """
    Given text, generate pairs of useful bigrams
    :returns : list of useful bigrams

    >> [a_b, c_d, ]
    """
    stopwords = open("data/stopwords", mode='r', encoding='utf8').read().splitlines()
    stopwords = sorted(stopwords)
    assert is_stopwords(stopwords, "yourselves")

    text = text.lower()
    text = text[:int(len(text) * read_percent)]

    words = word_tokenize(text)
    words = [word for word in words if not is_stopwords(stopwords, word)]
    print(len(words), " words")

    words = [stemmer.stem(word) for word in words]
    bigrams = nltk.bigrams(words)
    bigrams = ["_".join(bigram) for bigram in bigrams]

    print("Filtering from {} bigrams".format(len(bigrams)))
    bigrams_counter = Counter(bigrams).most_common(int(bigram_percent * len(bigrams)))

    useful_bigrams = []
    for bigram, count in bigrams_counter:
        a, b = bigram.split("_")
        if not (is_stopwords(stopwords, a) or is_stopwords(stopwords, b)):
            useful_bigrams.append(bigram)

    print("Selected a total of {} bigrams".format(len(useful_bigrams)))

    useful_bigrams = sorted(useful_bigrams)
    with open(save_filename, mode='w', encoding='utf8') as file:
        file.write("\n".join(useful_bigrams))

    return useful_bigrams


class Preprocess:
    def __init__(self, bigram_filename="data/useful_bigrams.txt", bigrams=False, vocab_size=None,
                 remove_stopwords=True):
        self.bigram_filename = bigram_filename
        self.vocab_size = vocab_size
        self.bigrams = bigrams
        self.stopwords = open("data/stopwords", mode='r', encoding='utf8').readlines()
        self.remove_stopwords = remove_stopwords

        self.useful_bigrams = []
        if os.path.exists(bigram_filename):
            self.useful_bigrams = open(bigram_filename, mode='r', encoding='utf8').read().splitlines()
            self.useful_bigrams = sorted(self.useful_bigrams)

    def build_vocab(self, text, stem=True):
        """
        :param vocab_size : size of vocabulary to consider other words will be replaced with UNK
            total_vocab_size = vocab_size + 1
            vocab_size of -1 means consider all the words
        :param text : text from which to build vocabulary
        type string
        """
        text = text.lower()
        self.words = word_tokenizer.tokenize(text)

        all_vocab = set(self.words)

        if stem:
            lemmatized_vocab = [stemmer.stem(word) for word in all_vocab]
            self.lemmatized_dict = dict(zip(all_vocab, lemmatized_vocab))
            self.words = [self.lemmatized_dict[word] for word in self.words]

        word_counter = Counter(self.words)

        vocab_size = self.vocab_size
        if not self.vocab_size:
            vocab_size = len(word_counter)
        vocab = word_counter.most_common(vocab_size)

        self.vocab = set([word for word, count in vocab])

    def preprocess(self, sentences):
        """
        :param bigrams: True or False. Should bigrams be generated or not?
        :param bigram_filename: filename containing the useful bigrams to consider
        :return: vocab, X, Y, label

        Steps:
            Split into sentences
            Split sentence into words
            Remove stopwords and lemmatize / stem words
            Generate ngrams

        "I need my credit card." -> "I need my credit card credit_card"
        """
        processed_sentences = []
        for sentence in sentences:
            sentence = word_tokenizer.tokenize(sentence)
            sentence = [word for word in sentence if not is_stopwords(self.stopwords, word)]

            sentence = [stemmer.stem(word) for word in sentence]

            if self.vocab_size:
                sentence = [word if word in self.vocab else 'unk' for word in sentence]

            if self.bigrams:
                bigrams = list(nltk.bigrams(sentence))
                bigrams = ["_".join(bigram) for bigram in bigrams]
                bigrams = [bigram for bigram in bigrams if is_useful_bigram(self.useful_bigrams, bigram)]
                sentence += bigrams

            sentence = " ".join(sentence)
            processed_sentences.append(sentence)

        return processed_sentences


if __name__ == '__main__':
    text = open("data/squad-base", mode='r', encoding='utf8').read().lower()
    generate_bigrams(text, save_filename='squad_bigrams', read_percent=1, bigram_percent=0.01)

    # text = open("data/base", mode='r', encoding='utf8').read().lower()
    # generate_bigrams(text, save_filename='base_bigrams', read_percent=0.05, bigram_percent=0.001)
