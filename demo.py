from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mutual_info_score
import numpy as np

corpus = [
    'This is the first document.',
    'Is this a first document?',
    'Is this a cat?',
    'And this is the third one.',
]

vectorizer = TfidfVectorizer().fit(corpus)
vectors = vectorizer.transform(corpus).todense()
vocabulary = vectorizer.vocabulary_
vocabulary = ((key, word) for word, key in vocabulary.items())
vocabulary = sorted(vocabulary, key=lambda x: x[0])

print(vocabulary)
print()

for sentence, vector in zip(corpus, vectors):
    print(sentence)
    print(np.round(vector, 2).ravel())
    print()

vec_a = vectors[0]
vec_b = vectors[1]
vec_c = vectors[2]

print("Cosine distances")
print(np.round(cosine_similarity(vec_a, vec_b)), 2)
print(np.round(cosine_similarity(vec_a, vec_c)), 2)
print(np.round(cosine_similarity(vec_b, vec_c)), 2)

vec_a = vec_a.flatten().tolist()[0]
vec_c = vec_c.flatten().tolist()[0]
vec_b = vec_b.flatten().tolist()[0]

print(np.round(mutual_info_score(vec_a, vec_b), 2))
print(np.round(mutual_info_score(vec_a, vec_c), 2))
print(np.round(mutual_info_score(vec_b, vec_c), 2))
