# Proposed Plan

## Binary Classification on Ethos HateSpeech Dataset

**!Task: Contains hate speech or not?!**

### Vectorizing with TFIDF:
1. maximum_words: [5000, 10000, 50000, 100000]
2. n_grams: [(1,1),(1,2),(1,5)]

### Run Grid Searches on:
1. Logistic Regression
2. RidgeClassificassion
3. SVM (linear & rbf)
4. Decision Tree
5. Random Forests
6. GradBoost
7. Naive Bayes
8. AdaBoost
9. BaggingClassifier (with max_samples &/|| max_features)

### text_to_secuenxe & Embeddings:
1. FastText
2. GloVe
3. GloVe + FastText
4. Bert

### Run Grid Searches on these Neurals:
1. LSTMs
2. BiLSTMs
3. CNNs
4. LSTMs + CNNs
5. LSTMs + BiLSTMs + CNNs
6. Full Connected
7. Full Connected + LSTMs + BiLSTMs + CNNs

## MultiClass Classification on Ethos HateSpeech Dataset

**Based on the previous best models do:**

1. Binary Relevance
2. Pairwise
3. Classifier Chains
4. Classifier Chains for some labels (maybe the 'isHate' label) and then binary relevance


## Metrics

**Metrics on Binary:**
1. Accuracy
2. F1 Macro
3. Precision Macro
4. Recall Macro
5. Sensitivity
6. Specificity

**Metrics on MultiLabel**
1. Accuracy, B Macro, B Micro
2. Hamming Loss
3. Precision, Recall, F1
4. Average Precision
