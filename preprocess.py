from collections import Counter
import os

import nltk
from nltk import WordPunctTokenizer, PunktSentenceTokenizer, PorterStemmer

sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = WordPunctTokenizer()
stemmer = PorterStemmer()


class Preprocess:
    def __init__(self, bigram_filename="data/useful_bigrams.txt", bigrams=False, vocab_size=None,
                 remove_stopwords=True):
        self.bigram_filename = bigram_filename
        self.vocab_size = vocab_size
        self.bigrams = bigrams
        self.stopwords = open("data/stopwords", mode='r', encoding='utf8').readlines()
        self.remove_stopwords = remove_stopwords
        if os.path.exists(bigram_filename):
            self.useful_bigrams = open(bigram_filename, mode='r', encoding='utf8').readlines()

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
            print("Stemmin")
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

            sentence = [stemmer.stem(word) for word in sentence]

            if self.vocab_size:
                sentence = [word if word in self.vocab else 'unk' for word in sentence]

            if self.bigrams:
                bigrams = list(nltk.bigrams(sentence))
                bigrams = ["_".join(bigram) for bigram in bigrams]
                bigrams = [bigram for bigram in bigrams if bigram in self.useful_bigrams]
                sentence += bigrams

            sentence = " ".join(sentence)
            processed_sentences.append(sentence)

        return processed_sentences

    def generate_bigrams(self, t=1, threshold=0.05):
        """
        Given text, generate pairs of useful bigrams

        Formula to generate bigram:
            is_valid_pair(a,b) = count(a, b) - threshold / (count(a) * count(b))

        :returns : list of useful bigrams

        >> [a_b, c_d, ]
        """
        eps = 1e-100

        lemmatized_vocab = [stemmer.stem(word) for word in self.vocab]
        lemmatized_dict = dict(zip(self.vocab, lemmatized_vocab))
        # print("Words ", words)
        words = [lemmatized_dict.get(word, 'unk') for word in self.words]
        # print("Lemmatized :", words)
        bigrams = list(nltk.bigrams(words))

        word_counter = Counter(words)
        bigrams_counter = Counter(bigrams)

        useful_bigrams = []
        for bigram, count in bigrams_counter.items():
            score = (bigrams_counter[bigram] - t) / (word_counter[bigram[0]] * word_counter[bigram[1]] + eps)
            if score > threshold:
                useful_bigrams.append(bigram)

        useful_bigrams = ["_".join(bigram) for bigram in useful_bigrams]
        self.useful_bigrams = sorted(useful_bigrams)

        with open(self.bigram_filename, mode='w', encoding='utf8') as file:
            file.write("\n".join(useful_bigrams))

        return self.useful_bigrams


if __name__ == '__main__':
    text = "What are the uses of credit card ? Can I get a credit card today? Ram is a player. He plays play cricket really well"
    p = Preprocess(bigrams=True, vocab_size=22)
    p.build_vocab(text, stem=True)
    print(p.vocab.__len__())
    p.generate_bigrams()
    ps = p.preprocess(text.split("?"))
    print(ps)
    pass
