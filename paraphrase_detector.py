import sklearn
import numpy as np
import os

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
        pass

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
        pass

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
    pass
