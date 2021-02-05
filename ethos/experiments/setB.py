"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#####################################################################
#                           Set B                                   #
#####################################################################
# In this set of experiments we will try AdaBoost, GraBoost and     #
# Bagging across a wide variety of parameters for each algorithm    #
# and test them via nested cross validation method.                 #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from utilities.preprocess import Preproccesor
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import time


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(tn+fp) > 0:
        speci = tn/(tn+fp)
        return speci
    return 0


def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    if(tp+fn) > 0:
        sensi = tp/(tp+fn)
        return sensi
    return 0


def nested_cross_val(pipe, parameters, X, y, name):
    scores = {}
    scores.setdefault('fit_time', [])
    scores.setdefault('score_time', [])
    scores.setdefault('test_F1', [])
    scores.setdefault('test_Precision', [])
    scores.setdefault('test_Recall', [])
    scores.setdefault('test_Accuracy', [])
    scores.setdefault('test_Specificity', [])
    scores.setdefault('test_Sensitivity', [])

    outer_cv = KFold(n_splits=10, shuffle=True, random_state=0)
    splits = outer_cv.split(X)
    for train_index, test_index in splits:
        X_trainO, X_testO = X[train_index], X[test_index]
        y_trainO, y_testO = y[train_index], y[test_index]
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
        clf = GridSearchCV(estimator=pipe, param_grid=parameters,
                           cv=inner_cv, n_jobs=22, verbose=1, scoring='f1')
        a = time.time()
        clf.fit(X_trainO, y_trainO)
        fit_time = time.time() - a
        a = time.time()
        y_preds = clf.predict(X_testO)
        score_time = time.time() - a
        scores['fit_time'].append(fit_time)
        scores['score_time'].append(score_time)
        scores['test_F1'].append(f1_score(y_testO, y_preds, average='macro'))
        scores['test_Precision'].append(
            precision_score(y_testO, y_preds, average='macro'))
        scores['test_Recall'].append(
            recall_score(y_testO, y_preds, average='macro'))
        scores['test_Accuracy'].append(accuracy_score(y_testO, y_preds))
        scores['test_Specificity'].append(specificity(y_testO, y_preds))
        scores['test_Sensitivity'].append(sensitivity(y_testO, y_preds))
    for k in scores:
        print(str(name)+" "+str(k)+": "+str(sum(scores[k])/10))
    f = open("../results/setB.txt", "a+")
    f.write("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format(str(name)[:7],
                                                                                str('%.4f' % (sum(scores['fit_time'])/10)), str('%.4f' % (
                                                                                    sum(scores['score_time'])/10)), str('%.4f' % (sum(scores['test_F1'])/10)),
                                                                                str('%.4f' % (sum(scores['test_Precision'])/10)), str('%.4f' % (
                                                                                    sum(scores['test_Recall'])/10)), str('%.4f' % (sum(scores['test_Accuracy'])/10)),
                                                                                str('%.4f' % (sum(scores['test_Specificity'])/10)), str('%.4f' % (sum(scores['test_Sensitivity'])/10))))
    f.close()
    print("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7}".format(str(name)[:7],
                                                                           str('%.4f' % (sum(scores['fit_time'])/10)), str('%.4f' % (
                                                                               sum(scores['score_time'])/10)), str('%.4f' % (sum(scores['test_F1'])/10)),
                                                                           str('%.4f' % (sum(scores['test_Precision'])/10)), str('%.4f' % (
                                                                               sum(scores['test_Recall'])/10)), str('%.4f' % (sum(scores['test_Accuracy'])/10)),
                                                                           str('%.4f' % (sum(scores['test_Specificity'])/10)), str('%.4f' % (sum(scores['test_Sensitivity'])/10))))


X, y = Preproccesor.load_data(True)
f = open("../results/setB.txt", "w+")
f.write("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format(
    'Method', 'fitTime', 'scoreTi', 'F1score', 'Precisi', 'Recall', 'Accurac', 'Specifi', 'Sensiti'))
f.write("=========================================================================\n")
f.close()
print("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7}".format('Method',
                                                                       'fitTime', 'scoreTi', 'F1score', 'Precisi', 'Recall', 'Accurac', 'Specifi', 'Sensiti'))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run AdaBoost
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
ada = AdaBoostClassifier()
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('ada', ada)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'ada__base_estimator':[None, DecisionTreeClassifier(max_depth=10), LogisticRegression(C=100)],
    'ada__n_estimators':[10, 50, 100, 300],
    'ada__learning_rate':[0.0001, 0.01, 0.5, 1]
}]
nested_cross_val(pipe, parameters, X, y, "AdaB")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run GradBoost
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
grad = GradientBoostingClassifier()
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('grad', grad)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'grad__learning_rate':[0.0001, 0.01, 0.1, 0.5, 1],
    'grad__n_estimators':[10, 50, 100, 300],
    'grad__subsample':[0.7, 0.85, 1],
    'grad__max_features':['sqrt', 'log2', None]
}]
nested_cross_val(pipe, parameters, X, y, "GradB")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run Bagging
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
bag = BaggingClassifier(n_jobs=-1)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('bag', bag)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'bag__base_estimator':[None, DecisionTreeClassifier(max_depth=10), LogisticRegression(C=100)],
    'bag__n_estimators':[10, 50, 100, 300],
    'bag__max_samples':[0.7, 0.85, 1],
    'bag__max_features':[0.5, 0.75, 1],
    'bag__bootstrap':[True, False]
}]
nested_cross_val(pipe, parameters, X, y, "Bagging")
