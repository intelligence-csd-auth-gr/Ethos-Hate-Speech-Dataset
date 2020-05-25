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
        text = [word for word in text if word not in stopwords.words('english')]
        wordnet_lemmatizer = WordNetLemmatizer()
        englishStemmer = SnowballStemmer("english", ignore_stopwords=True)
        text = [englishStemmer.stem(word) for word in text]
        text = [wordnet_lemmatizer.lemmatize(word) for word in text]
        # text = [lancaster.stem(word) for word in text]
        text = [word for word in text if word not in stopwords.words('english')]
    elif stops:
        text = [word for word in text if word not in stopwords.words('english')]
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

    def load_data(preprocessed = True):
        data = pd.read_csv("Ethos_Dataset.csv", delimiter=';')
        data = shuffle(data)

        XT = data['comment'].values
        X = []
        yT = data['isHate'].values #Add all the labels here :/
        y = []
        for yt in yT:
            if yt>=0.5:
                y.append(int(1))
            else:
                y.append(int(0))
        for x in XT:
            if preprocessed:
                X.append(my_clean(text=str(x), stops=False, stemming=True))
            else:
                X.append(x)
        return numpy.array(X),numpy.array(y)

    def load_multi_label_data(preprocessed = True):
        data = pd.read_csv("Ethos_Dataset.csv", delimiter=';')
        data = shuffle(data)

        XT = data['comment'].values
        X = []
        yT = data.loc[:,data.columns != 'comment'].values #Add all the labels here :/
        y = []
        for yt in yT:
            yi = []
            for i in yt:
                if i>=0.5:
                    yi.append(int(1))
                else:
                    yi.append(int(0))
            y.append(yi)
        for x in XT:
            if preprocessed:
                X.append(my_clean(text=str(x), stops=False, stemming=True))
            else:
                X.append(x)
        return numpy.array(X),numpy.array(yT),numpy.array(y)

