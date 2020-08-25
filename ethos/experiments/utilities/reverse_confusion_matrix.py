"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#####################################################################
#                        Reverse Script                             #
#####################################################################
# This script given the sensitivity and accuracy scores, as well    #
# as the total number of samples, and the number of positive labels #
# it calculates and returns the confusion matrix.                   #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.classification import _prf_divide
from preprocess import Preproccesor
import numpy as np

def my_confusion_matrix(se,acc,all,pos):
    tp = se * pos
    tn = (acc * all) - tp
    fn = tp * ((1 / se) - 1)
    fp = all - tp - tn - fn
    return tn, fp, fn, tp

se = 0.7843
acc = 0.7664
tn, fp, fn, tp = my_confusion_matrix(se,acc,998,433)
print(tn, fp, fn, tp)
tp_sum = np.array([tn,tp])
pred_sum = tp_sum + np.array([fp,fn])
true_sum = tp_sum + np.array([fn,fp])
beta2 = beta = 1

precision = _prf_divide(tp_sum, pred_sum, 'precision', 'predicted', None, ('precision','recall','f-score'))
recall = _prf_divide(tp_sum, true_sum, 'recall', 'true', None,  ('precision','recall','f-score'))

denom = beta2 * precision + recall
denom[denom == 0.] = 1  # avoid division by 0
f_score = (1 + beta2) * precision * recall / denom
print(f_score[1],precision[1],recall[1])