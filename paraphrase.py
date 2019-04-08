import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.stats import entropy
from sklearn.svm import SVC

from dataloader import dataLoader
from preprocess import Preprocess


class ParaphraseDetector:
    def __init__(self, distance='cosine', use_bigrams=True, stem=True):
        """
        :param distance : cosine, kld
        Steps:
        For given filenames if processed_filename does not exist,
            load unprocessed data and process it
            Save processed data so that you don't have to preprocess all the time
        else load processed data
        """
        distance_metrics = {
            'cosine': cosine_similarity,
            'kld': self.kld
        }
        self.distance_fn = distance_metrics[distance]

        train_filename = "data/msr_paraphrase_train.txt"
        test_filename = "data/msr_paraphrase_test.txt"
        self.train_labels, self.train_sentence1, self.train_sentence2 = dataLoader(train_filename)
        self.test_labels, self.test_sentence1, self.test_sentence2 = dataLoader(test_filename)
        assert len(self.train_sentence1) == len(self.train_sentence2)
        assert len(self.test_sentence1) == len(self.test_sentence2)

        self.svd = TruncatedSVD(n_components=8000)
        self.svm = SVC(gamma='auto')

        text = self.train_sentence2 + self.train_sentence1
        text = " ".join(text)

        self.preprocess = Preprocess(bigrams=True)
        self.preprocess.build_vocab(text, stem=True)
        print("Vocab : ", len(self.preprocess.vocab))
        self.preprocess.generate_bigrams(t=2, threshold=0.05)

        self.train_sentence1 = self.preprocess.preprocess(self.train_sentence1)
        self.train_sentence2 = self.preprocess.preprocess(self.train_sentence2)

        self.test_sentence1 = self.preprocess.preprocess(self.test_sentence1)
        self.test_sentence2 = self.preprocess.preprocess(self.test_sentence2)

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
        """
        corpus = self.train_sentence1 + self.train_sentence2
        # corpus = [
        #     'This is the first document.',
        #     'This document is the second document.',
        #     'And this is the third one.',
        #     'Is this the first document?']

        self.vectorizer = TfidfVectorizer().fit(corpus)  # It requires list of strings (sentences) not list of list
        print("tfidf Vocab  size : ", self.vectorizer.vocabulary_.__len__())

        self.train_tfidf_1 = self.vectorizer.transform(self.train_sentence1).todense()
        self.train_tfidf_2 = self.vectorizer.transform(self.train_sentence2).todense()

        self.test_tfidf_1 = self.vectorizer.transform(self.test_sentence1).todense()
        self.test_tfidf_2 = self.vectorizer.transform(self.test_sentence2).todense()

        accuracy_hdim = self.accuracy(self.train_tfidf_1, self.train_tfidf_2, self.train_labels, 0.625)
        print("hdim accuracy : ", "%.2f" % accuracy_hdim)

        self.tfidf_1_ldim = self.svd.fit_transform(self.train_tfidf_1)
        self.tfidf_2_ldim = self.svd.fit_transform(self.train_tfidf_2)

        self.tfidf_1_ldim = [np.asarray(vec).reshape(1, -1) for vec in self.tfidf_1_ldim]
        self.tfidf_2_ldim = [np.asarray(vec).reshape(1, -1) for vec in self.tfidf_2_ldim]

        self.tfidf_1_ldim = np.asarray(self.tfidf_1_ldim)
        self.tfidf_2_ldim = np.asarray(self.tfidf_2_ldim)

        accuracy_ldim = self.accuracy(self.tfidf_1_ldim, self.tfidf_2_ldim, self.train_labels)

        print("ldim accuracy : ", "%.2f" % accuracy_ldim, 0.4)

    def train(self, train_vec1, train_vec2, train_labels, test_vec1, test_vec2, test_labels):
        """ Train SVM and find it's accuracy """
        train_features = np.concatenate((train_vec1, train_vec2), axis=1)
        test_features = np.concatenate((test_vec1, test_vec2), axis=1)
        self.svm.fit(train_features, train_labels)
        accuracy = self.svm.score(test_features, test_labels)
        print("Accuracy on test set : ", accuracy)

    def accuracy(self, tfidf_1, tfidf_2, labels, thresh=0.6):
        """ Can use different kinds of distance function """
        scores = [self.distance_fn(vec_1, vec_2).item() for vec_1, vec_2 in zip(tfidf_1, tfidf_2)]
        false_score = [score for score, label in zip(scores, labels) if label == 0]
        true_score = [score for score, label in zip(scores, labels) if label == 1]

        false_score = sorted(false_score)[:(int(1 * len(false_score)))]
        true_score = sorted(true_score, reverse=True)[:(int(1 * len(true_score)))]
        auto_thresh = (np.average(false_score) + np.average(true_score)) / 2

        print("False average : ", np.average(false_score))
        print("True average : ", np.average(true_score))
        print("Auto Thresh : ", auto_thresh)

        predicted = [score >= thresh for score in scores]
        accuracy = [pred == label for pred, label in zip(predicted, labels)]
        return sum(accuracy) / len(accuracy)

    @staticmethod
    def kld(p, q):
        """
        :param p:
        :param q:
        :return: returns the KL divergence distance two probability distributions p and q
        """
        eps = 0
        p += eps
        q += eps
        kl = 1 - sum(entropy(p, q))
        return kl

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
    cp = ParaphraseDetector(distance='cosine')
    cp.compare()

    # print("Entropy: ", entropy([0.5, 0.5]))
    # print("Entropy: ", entropy([1, 0]))
    # print("Entropy: ", mutual_info_score([0, 1], [1, 0]))
