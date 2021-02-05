"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#####################################################################
#                           Set E                                   #
#####################################################################
# In this set of experiments we will try classic multi label        #
# algorithms to find the best classifiers.                          #
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, make_scorer, fbeta_score, multilabel_confusion_matrix,\
    average_precision_score, precision_score, recall_score
from skmultilearn.adapt import MLkNN, MLTSVM, MLARAM
from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance
from utilities.preprocess import Preproccesor
import nltk
import warnings


def average_precision_wrapper(y, y_pred, view):
    return average_precision_score(y, y_pred.toarray(), average=view)


hamm_scorer = make_scorer(hamming_loss, greater_is_better=False)
ftwo_scorer = make_scorer(fbeta_score, beta=2)

nltk.download('wordnet')
nltk.download('stopwords')
X, yt, y = Preproccesor.load_multi_label_data(
    True)  # yt has continuous data, y has binary
label_names = ["violence", "directed_vs_generalized", "gender", "race",
               "national_origin", "disability", "religion", "sexual_orientation"]

warnings.filterwarnings('ignore')

f = open("../results/setE.txt", "w+")
f.write("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format('Method',
        'F1_Ex', 'F1_Ma', 'F1_Mi',
        'Pre_Ex', 'Pre_Ma', 'Pre_Mi',
        'Re_Ex', 'Re_Ma', 'Re_Mi',
        'AP_Ma', 'AP_Mi', 'Accurac', 'Hamm'))
f.write("================================================================================================================\n")
f.close()
print("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} ".format('Method',
        'F1_Ex', 'F1_Ma', 'F1_Mi',
        'Pre_Ex', 'Pre_Ma', 'Pre_Mi',
        'Re_Ex', 'Re_Ma', 'Re_Mi',
        'AP_Ma', 'AP_Mi', 'Accurac', 'Hamm'))


def nested_cross_val(pipe, parameters, X, y, name):
    scores = {}
    scores.setdefault('test_F1_example', [])
    scores.setdefault('test_F1_macro', [])
    scores.setdefault('test_F1_micro', [])
    scores.setdefault('test_precision_example', [])
    scores.setdefault('test_precision_macro', [])
    scores.setdefault('test_precision_micro', [])
    scores.setdefault('test_recall_example', [])
    scores.setdefault('test_recall_macro', [])
    scores.setdefault('test_recall_micro', [])
    scores.setdefault('test_average_precision_macro', [])
    scores.setdefault('test_average_precision_micro', [])
    scores.setdefault('test_Accuracy', [])
    scores.setdefault('test_Hamm', [])
    cm = []
    outer_cv = MultilabelStratifiedKFold(n_splits=10, random_state=0)
    splits = outer_cv.split(X, y)
    for train_index, test_index in splits:
        X_trainO, X_testO = X[train_index], X[test_index]
        y_trainO, y_testO = y[train_index], y[test_index]
        inner_cv = MultilabelStratifiedKFold(n_splits=10, random_state=0)
        clf = GridSearchCV(estimator=pipe, param_grid=parameters,
                           cv=inner_cv, n_jobs=-1, verbose=1, scoring=hamm_scorer)
        clf.fit(X_trainO, y_trainO)
        y_preds = clf.predict(X_testO)
        cm.append(multilabel_confusion_matrix(y_testO, y_preds))
        scores['test_F1_example'].append(
            f1_score(y_testO, y_preds, average='samples'))
        scores['test_F1_macro'].append(
            f1_score(y_testO, y_preds, average='macro'))
        scores['test_F1_micro'].append(
            f1_score(y_testO, y_preds, average='micro'))
        scores['test_precision_example'].append(
            precision_score(y_testO, y_preds, average='samples'))
        scores['test_precision_macro'].append(
            precision_score(y_testO, y_preds, average='macro'))
        scores['test_precision_micro'].append(
            precision_score(y_testO, y_preds, average='micro'))
        scores['test_recall_example'].append(
            recall_score(y_testO, y_preds, average='samples'))
        scores['test_recall_macro'].append(
            recall_score(y_testO, y_preds, average='macro'))
        scores['test_recall_micro'].append(
            recall_score(y_testO, y_preds, average='micro'))
        if name == "MLARAM":
            scores['test_average_precision_macro'].append(
                average_precision_score(y_testO, y_preds, average='macro'))
            scores['test_average_precision_micro'].append(
                average_precision_score(y_testO, y_preds, average='micro'))
        else:
            scores['test_average_precision_macro'].append(
                average_precision_wrapper(y_testO, y_preds, 'macro'))
            scores['test_average_precision_micro'].append(
                average_precision_wrapper(y_testO, y_preds, 'micro'))
        scores['test_Accuracy'].append(accuracy_score(y_testO, y_preds))
        scores['test_Hamm'].append(hamming_loss(y_testO, y_preds))
    cmt = cm[0]
    for ra in range(1, len(cm)):
        cmt = cmt + ra
    cmt = cmt/10
    print(cmt)
    f = open("../results/setE.txt", "a+")
    f.write("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format(str(name)[:7],
            str('%.4f' % (
                sum(scores['test_F1_example'])/10)),
            str('%.4f' % (
                sum(scores['test_F1_macro'])/10)),
            str('%.4f' % (
                sum(scores['test_F1_micro']) / 10)),
            str('%.4f' % (
                sum(scores['test_precision_example']) / 10)),
            str('%.4f' % (
                sum(scores['test_precision_macro']) / 10)),
            str('%.4f' % (
                sum(scores['test_precision_micro']) / 10)),
            str('%.4f' % (
                sum(scores['test_recall_example']) / 10)),
            str('%.4f' % (
                sum(scores['test_recall_macro']) / 10)),
            str('%.4f' % (
                sum(scores['test_recall_micro']) / 10)),
            str('%.4f' % (
                sum(scores['test_average_precision_macro'])/10)),
            str('%.4f' % (
                sum(scores['test_average_precision_micro'])/10)),
            str('%.4f' % (
                sum(scores['test_Accuracy'])/10)),
            str('%.4f' % (sum(scores['test_Hamm'])/10))))
    f.close()
    print("{:<7} | {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} {:<7} \n".format(str(name)[:7],
            str('%.4f' % (
                sum(scores['test_F1_example'])/10)),
            str('%.4f' % (
                sum(scores['test_F1_macro'])/10)),
            str('%.4f' % (
                sum(scores['test_F1_micro']) / 10)),
            str('%.4f' % (
                sum(scores['test_precision_example']) / 10)),
            str('%.4f' % (
                sum(scores['test_precision_macro']) / 10)),
            str('%.4f' % (
                sum(scores['test_precision_micro']) / 10)),
            str('%.4f' % (
                sum(scores['test_recall_example']) / 10)),
            str('%.4f' % (
                sum(scores['test_recall_macro']) / 10)),
            str('%.4f' % (
                sum(scores['test_recall_micro']) / 10)),
            str('%.4f' % (
                sum(scores['test_average_precision_macro'])/10)),
            str('%.4f' % (
                sum(scores['test_average_precision_micro'])/10)),
            str('%.4f' % (
                sum(scores['test_Accuracy'])/10)),
            str('%.4f' % (sum(scores['test_Hamm'])/10))))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run MLkNN
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
mlknn = MLkNN()
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('MLkNN', mlknn)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'MLkNN__k':[1, 3, 10, 20, 50],
    'MLkNN__s': [0.5, 0.7, 1.0]
}]
nested_cross_val(pipe, parameters, X, y, "MLkNN")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run MLTSVM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
mltsvm = MLTSVM(max_iteration=1000)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('MLTSVM', mltsvm)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'MLTSVM__c_k': [2**i for i in range(-5, 10, 2)]
}]
nested_cross_val(pipe, parameters, X, y, "MLTSVM")


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run MLARAM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
mlaram = MLARAM()
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('MLARAM', mlaram)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'MLARAM__vigilance': [0.8, 0.9, 0.999],
    'MLARAM__threshold': [0.01, 0.1]
}]
nested_cross_val(pipe, parameters, X, y, "MLARAM")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run Binary Relevance
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
br = BinaryRelevance(require_dense=[False, True])
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('br', br)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'br__classifier':[LogisticRegression(C=1000), LogisticRegression(C=1000, solver='newton-cg'), SVC(C=1000), LogisticRegression(C=100), SVC(C=100), LogisticRegression(C=10), SVC(C=10), DecisionTreeClassifier(), RandomForestClassifier()]
}]
nested_cross_val(pipe, parameters, X, y, "BinRel")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Run ClassifierChains
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
clasChain = ClassifierChain(
    require_dense=[False, True]
)
vec = TfidfVectorizer(analyzer='word')
pipe = Pipeline(steps=[('vec', vec), ('ch', clasChain)])
parameters = [{
    'vec__ngram_range': [(1, 1), (1, 2), (1, 5)],
    'vec__max_features':[5000, 10000, 50000, 100000],
    'vec__stop_words':['english', None],
    'ch__classifier':[LogisticRegression(C=1000), SVC(C=1000), LogisticRegression(C=100), SVC(C=100), LogisticRegression(C=10), SVC(C=10), DecisionTreeClassifier(), RandomForestClassifier()]
}]
nested_cross_val(pipe, parameters, X, y, "ClassifierChains")