import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import TruncatedSVD
from scipy.special import kl_div
from scipy.stats import entropy
from sklearn.svm import SVC

from dataloader import dataLoader


# Does this require cross validation ?
# Have a function to measure performance - accuracy depending on threshold

class ComparePerformance:
    def __init__(self, strategy='tfidf', distance='cosine', use_bigrams=True):
        """
        :param strategy: 'tfidf' or 'tfkld'
        :param distance : cosine, kld
        Steps:
        For given filenames if processed_filename does not exist,
            load unprocessed data and process it
            Save processed data so that you don't have to preprocess all the time
        else load processed data
        """
        distance_metrics = {
            'cosine': cosine_similarity
        }
        self.distance_fn = distance_metrics[distance]

        train_filename = "data/msr_paraphrase_train.txt"
        test_filename = "data/msr_paraphrase_test.txt"
        self.train_labels, self.train_sentence1, self.train_sentence2 = dataLoader(train_filename)
        self.test_labels, self.test_sentence1, self.test_sentence2 = dataLoader(test_filename)
        assert len(self.train_sentence1) == len(self.train_sentence2)
        assert len(self.test_sentence1) == len(self.test_sentence2)

        self.svd = TruncatedSVD(n_components=4)
        self.svm = SVC(gamma='auto')

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
        corpus = self.train_sentence1
        # corpus = [
        #     'This is the first document.',
        #     'This document is the second document.',
        #     'And this is the third one.',
        #     'Is this the first document?']

        self.vectorizer = TfidfVectorizer().fit(corpus)  # It requires list of strings (sentences) not list of list

        self.train_tfidf_1 = self.vectorizer.transform(corpus).todense()
        self.train_tfidf_2 = self.vectorizer.transform(self.train_sentence2).todense()

        self.test_tfidf_1 = self.vectorizer.transform(self.test_sentence1).todense()
        self.test_tfidf_2 = self.vectorizer.transform(self.test_sentence2).todense()

        accuracy_hdim = self.accuracy(self.train_tfidf_1, self.train_tfidf_2, self.train_labels)
        print("hdim accuracy : ", "%.2f" % accuracy_hdim)

        self.tfidf_1_ldim = self.svd.fit_transform(self.train_tfidf_1)
        self.tfidf_2_ldim = self.svd.fit_transform(self.train_tfidf_2)

        self.tfidf_1_ldim = [np.asarray(vec).reshape(1, -1) for vec in self.tfidf_1_ldim]
        self.tfidf_2_ldim = [np.asarray(vec).reshape(1, -1) for vec in self.tfidf_2_ldim]

        self.tfidf_1_ldim = np.asarray(self.tfidf_1_ldim)
        self.tfidf_2_ldim = np.asarray(self.tfidf_2_ldim)

        accuracy_ldim = self.accuracy(self.tfidf_1_ldim, self.tfidf_2_ldim, self.train_labels)

        print("ldim accuracy : ", "%.2f" % accuracy_ldim)

        # print(cosine_similarity(self.train_tfidf_1[0], self.train_tfidf_1[3]))
        # print(mutual_info_score(self.train_tfidf_1[0].todense(), self.train_tfidf_1[3].todense()))

        # print("Cosine : ", cosine_similarity(self.train_tfidf_1[0], self.train_tfidf_1[3]))
        # print("KL Divergence : ", kl_div(self.reduced_tfidf[0], self.reduced_tfidf[3]))
        # print(self.reduced_tfidf[3])
        # print("Entropy: ", entropy(self.reduced_tfidf[0], self.reduced_tfidf[3]))

    def train(self, train_vec1, train_vec2, train_labels, test_vec1, test_vec2, test_labels):
        """ Train SVM and find it's accuracy """
        train_features = np.concatenate((train_vec1, train_vec2), axis=1)
        test_features = np.concatenate((test_vec1, test_vec2), axis=1)
        self.svm.fit(train_features, train_labels)
        accuracy = self.svm.score(test_features, test_labels)
        print("Accuracy on test set : ", accuracy)

    def accuracy(self, tfidf_1, tfidf_2, labels, thresh=0.6):
        """ Can use different kinds of distance function """
        predicted = [self.distance_fn(vec_1, vec_2).item() > thresh for vec_1, vec_2 in zip(tfidf_1, tfidf_2)]
        accuracy = [pred == label for pred, label in zip(predicted, labels)]
        return sum(accuracy) / len(accuracy)

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
    cp = ComparePerformance()
    cp.compare()

    # print("Entropy: ", entropy([0.5, 0.5]))
    # print("Entropy: ", entropy([1, 0]))
    # print("Entropy: ", mutual_info_score([0, 1], [1, 0]))
