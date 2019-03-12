import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import TruncatedSVD
from scipy.special import kl_div
from scipy.stats import entropy

from dataloader import dataLoader

# Does this require cross validation ?
# Have a function to measure performance - accuracy depending on threshold

class ComparePerformance:
    def __init__(self, filename, strategy='tfidf', distance='cosine', use_bigrams=True):
        """
        :param strategy: 'tfidf' or 'tfkld'
        :param distance : cosine, kld
        Steps:
        For given filenames if processed_filename does not exist,
            load unprocessed data and process it
            Save processed data so that you don't have to preprocess all the time
        else load processed data
        """
        self.label, self.sx, self.sy = dataLoader(filename)
        self.svd = TruncatedSVD(n_components=4)

    def compare(self):
        """
        Algorithm / Step:
            For a given strategy (tfidf or tfkld)
            Generate tf-idf matrix
            Find accuracy on paraphrase corpus
            Do low rank matrix approximation like Truncated SVD
            Find accuracy on paraphrase corpus
            Find delta accuracy
        :return:

        Call vectors of sentences in db as dbVector
        Call vectors of user query as queryVector
        """
        corpus = self.sx
        corpus = [
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?']

        # It requires list of strings (sentences) not list of list
        self.vectorizer = TfidfVectorizer()
        self.tfidf = self.vectorizer.fit_transform(corpus)
        # print(cosine_similarity(self.tfidf[0], self.tfidf[3]))
        # print(mutual_info_score(self.tfidf[0].todense(), self.tfidf[3].todense()))

        self.reduced_tfidf = self.svd.fit_transform(self.tfidf)
        print("Cosine : ", cosine_similarity(self.tfidf[0], self.tfidf[3]))
        print("KL Divergence : ", kl_div(self.reduced_tfidf[0], self.reduced_tfidf[3]))
        print(self.reduced_tfidf[3])
        print("Entropy: ", entropy(self.reduced_tfidf[0], self.reduced_tfidf[3]))

    @staticmethod
    def accuracy():
        pass

    def inference(self, query):
        """
        For given user query, find the closest match in Y.
        :param query: user query
        :return:

        Steps:
            1. Preprocess
            2. Transform words into TF-IDF vector
            3. Compute KLD or Cosine with every setence in Y
            4. Return best matching sentence
        """
        pass


if __name__ == '__main__':
    filename = "data/msr_paraphrase_test.txt"
    # cp = ComparePerformance(filename=filename)
    # cp.compare()

    print("Entropy: ", entropy([0.5, 0.5]))
    print("Entropy: ", entropy([1, 0]))
    print("Entropy: ", mutual_info_score([0, 1], [1, 0]))
