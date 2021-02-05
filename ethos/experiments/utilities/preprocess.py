"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#####################################################################
#                       Preprocess Script                           #
#####################################################################
# This script preprocesses the data.                                #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy
import numpy as np
from sklearn.utils import shuffle


def my_clean(text, stops=False, stemming=False):
    text = str(text)
    text = re.sub(r" US ", " american ", text)
    text = text.lower().split()
    text = " ".join(text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"don't", "do not ", text)
    text = re.sub(r"aren't", "are not ", text)
    text = re.sub(r"isn't", "is not ", text)
    text = re.sub(r"%", " percent ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"doesn't", "does not ", text)
    text = re.sub(r"he's", "he is ", text)
    text = re.sub(r"she's", "she is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lower().split()
    text = [w for w in text if len(w) >= 2]
    if stemming and stops:
        text = [
            word for word in text if word not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
        text = [englishStemmer.stem(word) for word in text]
        text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        text = [
            word for word in text if word not in stopwords.words('english')]
    elif stops:
        text = [
            word for word in text if word not in stopwords.words('english')]
    elif stemming:
        wordnet_lemmatizer = WordNetLemmatizer()
        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
        text = [englishStemmer.stem(word) for word in text]
        text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    text = " ".join(text)
    return text


class Preproccesor:

    def __init__(self):
        """Init function
        """

    def load_data(preprocessed=True, stemming_a=True):
        data = pd.read_csv(
            "../ethos_data/Ethos_Dataset_Binary.csv", delimiter=';')
        np.random.seed(2000)
        data = data.iloc[np.random.permutation(len(data))]
        XT = data['comment'].values
        X = []
        yT = data['isHate'].values
        y = []
        for yt in yT:
            if yt >= 0.5:
                y.append(int(1))
            else:
                y.append(int(0))
        for x in XT:
            if preprocessed:
                X.append(my_clean(text=str(x), stops=False, stemming=stemming_a))
            else:
                X.append(x)
        return numpy.array(X), numpy.array(y)

    def load_multi_label_data(preprocessed=True, stemming_a=True):
        data = pd.read_csv(
            "../ethos_data/Ethos_Dataset_Multi_Label.csv", delimiter=';')

        XT = data['comment'].values
        X = []
        # Add all the labels here 
        yT = data.loc[:, data.columns != 'comment'].values
        y = []
        for yt in yT:
            yi = []
            for i in yt:
                if i >= 0.5:
                    yi.append(int(1))
                else:
                    yi.append(int(0))
            y.append(yi)
        for x in XT:
            if preprocessed:
                X.append(my_clean(text=str(x), stops=False, stemming=stemming_a))
            else:
                X.append(x)
        return numpy.array(X), numpy.array(yT), numpy.array(y)

    def load_external_data(preprocessed=True, stemming_a=True):
        """
        @inproceedings{hateTweets,
          title = {Automated Hate Speech Detection and the Problem of Offensive Language},
          author = {Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar}, 
          booktitle = {Proceedings of the 11th International AAAI Conference on Web and Social Media},
          series = {ICWSM '17},
          year = {2017},
          publisher = {{AAAI} Press},
          address = {Montreal, Canada},  
          pages = {512-515}
        }
        """
        data = pd.read_csv("../hs_data/hate-speech-and-offensive-language.csv")
        X = []
        y = []
        XT = data['tweet'].values
        yT = data['class'].values
        yT = [0 if (i == 1 or i == 2) else 1 for i in yT]
        c = 0
        for x in XT:
            if preprocessed:
                X.append(my_clean(text=str(x), stops=False, stemming=stemming_a))
                y.append(yT[c])
            else:
                X.append(x)
                y.append(yT[c])
            c = c + 1
        return numpy.array(X), numpy.array(y)

    def load_mlma(preprocessed=True, stemming_a=True):
        """
        @inproceedings{DBLP:conf/emnlp/OusidhoumLZSY19,
          author    = {Nedjma Ousidhoum and
                       Zizheng Lin and
                       Hongming Zhang and
                       Yangqiu Song and
                       Dit{-}Yan Yeung},
          title     = {Multilingual and Multi-Aspect Hate Speech Analysis},
          booktitle = {{EMNLP-IJCNLP} 2019,
                       November 3-7, 2019},
          pages     = {4674--4683},
          address = {Hong Kong, China},
          publisher = {Association for Computational Linguistics},
          year      = {2019},
          url       = {https://doi.org/10.18653/v1/D19-1474},
          doi       = {10.18653/v1/D19-1474},
          timestamp = {Thu, 12 Dec 2019 13:23:50 +0100},
          biburl    = {https://dblp.org/rec/conf/emnlp/OusidhoumLZSY19.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
        }
        """
        data = pd.read_csv("../hs_data//en_dataset_with_stop_words.csv")
        X = []
        XT = data['tweet'].values
        c = 0
        for x in XT:
            if preprocessed:
                text = my_clean(text=str(x), stops=False, stemming=stemming_a)
                while 'user' in text:
                    text = text.replace('user', '')
                X.append(text)
            else:
                X.append(x)
            c = c + 1
        return numpy.array(X), data['sentiment'].values, data['directness'].values, data['annotator_sentiment'].values, data['target'].values, data['group'].values, ['tweet', 'sentiment', 'directness', 'annotator_sentiment', 'target', 'group']
