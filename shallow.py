import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from preproc import load_data, tokenize

parser = argparse.ArgumentParser()
parser.add_argument("--classifier", type=str, default='lr')

if __name__ == "__main__":

    train_df = load_data('train')
    valid_df = load_data('valid')

    x_train = train_df['text']
    y_train = train_df['stars']

    x_valid = valid_df['text']
    y_valid = valid_df['stars']

    tfidf = TfidfVectorizer(tokenizer=tokenize)
    tfidf.fit(x_train)

    args = parser.parse_args()
    if 'lr' == args.classifier:
        print("using single logistic regression")
        clf = LogisticRegression()
        clf.fit(tfidf.transform(x_train), y_train)
        y_pred = clf.predict(tfidf.transform(x_valid))

    if 'svm' == args.classifier:
        print("using single SVM")
        clf = SVC()
        clf.fit(tfidf.transform(x_train), y_train)
        y_pred = clf.predict(tfidf.transform(x_valid))

    elif 'adalr' == args.classifier:
        print("using Adaboost logistic regression")
        clf = AdaBoostClassifier(base_estimator=LogisticRegression(),
                                 n_estimators=1000, random_state=0)
        clf.fit(tfidf.transform(x_train), y_train)
        y_pred = clf.predict(tfidf.transform(x_valid))

    elif 'bagsvm' == args.classifier:
        print("using bagging SVM")
        clf = BaggingClassifier(base_estimator=SVC(),
                                n_estimators=50, random_state=0)
        clf.fit(tfidf.transform(x_train), y_train)
        y_pred = clf.predict(tfidf.transform(x_valid))

    print(classification_report(y_valid, y_pred))
    print("\n\n")
    print(confusion_matrix(y_valid, y_pred))
    print('accuracy', np.mean(y_valid == y_pred))