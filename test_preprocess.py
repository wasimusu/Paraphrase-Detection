import unittest
from preprocess import is_stopwords


class TestBisection(unittest.TestCase):
    def test_bisect(self):
        stopwords = open("data/stopwords", mode='r', encoding='utf8').read().splitlines()
        stopwords = sorted(stopwords)

        assert is_stopwords(stopwords, 'am')
        assert is_stopwords(stopwords, '?')
        assert is_stopwords(stopwords, "'")
        assert is_stopwords(stopwords, "yourselves")

if __name__ == '__main__':
    TestBisection()