# Paraphrase-Detection
Implementing various measures of paraphrase detection on Microsoft Paraphrase Corpus and checking their performance on original high dimension TF-IDF matrix and it's low dimension approximation

If two sentences mean exactly the same thing they are a paraphrase of each other.

The two sentences below are paraphrase of each other.
- The increase reflects lower credit losses and favorable interest rates.	
- The gain came as a result of fewer credit losses and lower interest rates.

The two sentences below are not paraphrase of each other.
- Hong Kong was flat, Australia , Singapore and South Korea lost 0.2-0.4 percent.	
- Australia was flat, Singapore was down 0.3 percent by midday and South Korea added 0.2 percent.

### Sentence/Paragraph representation
[TF-IDF](https://blog.christianperone.com/2011/09/machine-learning-text-feature-extraction-tf-idf-part-i/) (Term Frequency Inverse Document Frequency) is used to each represent sentence.

Given that we have three sentences in our corpus, the TF-IDF representation would be:
- This is the first document.
- Is this a first document?
- Is this a cat?

Key: (0, 'and'), (1, 'cat'), (2, 'document'), (3, 'first'), (4, 'is'), (5, 'one'), (6, 'the'), (7, 'third'), (8, 'this')

**Note:** The more the text, the better TF-IDF learns to assign appropriate weight to words.

The value at index 0 represents the weight of 'and'. The value at index 2 represents the weight of 'docuemnt' and so on.

The vector representation for the 
```
- This is the first document. [0.   0.   0.51 0.51 0.34 0.   0.51 0.   0.34]
- Is this a first document?   [0.   0.   0.59 0.59 0.39 0.   0.   0.   0.39]
- Is this a cat?              [0.   0.8  0.   0.   0.42 0.   0.   0.   0.42]
```

Now that we have a vector representation of sentence, we can use various paradigms like supervised classification or
just simple thresholding for cosines of two vectors to determine whether two sentences are paraphrase or not.

Classification results
- I used cosine distances to determine whether two sentences are paraphrase or not. If the value of cosine distance is greater than some threshold, then two sentences are paraphrase.
- I used the train set to compute a threshold which results best classification performance on train set. 

The original dimension is 13000 words. We discarded 5000 least used words and then applied dimension reduction (Truncated SVD):
After discarding 5000 words, we have 8000 words or dimensions which is our base case.

The result of dimension reduction is shown below:

 | Reduced Dimension           | Accuracy  | Score |
 |:-------------:| -----:|--------:|
 | 6000 | 72.22 | 80.36 |
 | 4000 | 72.10 | 80.29 |
 | 2000 | 72.19 | 80.53 |
 | 1000 | 71.29 | 80.18 |
 
 For comparison, SVM (Support Vector Machine) classifier resulted in 66.4% accuracy.

### Some observations
- Discarding rare words results in better vector representation for sentences.
- Performance does not decrease due to dimension reduction, computation time and memory consumption sure does.
- Classification performance improved after using TruncatedSVD for dimension reduction.
