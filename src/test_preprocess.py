import unittest
from src.preprocess import is_stopwords
from src.paraphrase import kld
import numpy as np


class TestBisection(unittest.TestCase):
    def test_bisect(self):
        stopwords = open("../data/stopwords", mode='r', encoding='utf8').read().splitlines()
        stopwords = sorted(stopwords)

        assert is_stopwords(stopwords, 'am')
        assert is_stopwords(stopwords, '?')
        assert is_stopwords(stopwords, "'")
        assert is_stopwords(stopwords, "yourselves")

    def test_kld(self):
        p = np.asarray([0.36, 0.48, 0.16, 0.16])
        q = np.asarray([0.333, 0.333, 0.333, 0.333])
        r = np.asarray([0, 0.1, 0.9, 0.1])
        kl1 = kld(p, p)
        # kl2 = kld(q, p)
        kl2 = kld(q, r)
        print(kl1, kl2)


if __name__ == '__main__':
    TestBisection()
