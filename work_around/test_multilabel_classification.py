from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pandas as pd

def read_csv(FILE_PATH):
    return pd.read_csv(FILE_PATH)


if __name__ == '__main__':
    train_df = read_csv('../data/train.csv')[:1000]

    corpus = train_df['comment_text']
    classes = []

    for row in train_df.iterrows():
        if row[1]['toxic'] is not 0:
            classes.append('toxic')
        elif row[1]['severe_toxic'] is not 0:
            classes.append('severe_toxic')
        elif row[1]['obscene'] is not 0:
            classes.append('obscene')
        elif row[1]['threat'] is not 0:
            classes.append('threat')
        elif row[1]['insult'] is not 0:
            classes.append('insult')
        elif row[1]['identity_hate'] is not 0:
            classes.append('identity_hate')


    print(classes)
    test = ["Andrew is a stupid guy"]
    tvect = TfidfVectorizer(min_df=1, max_df=1)

    X = tvect.fit_transform(corpus)

    classifier = LogisticRegression()
    classifier.fit(X, classes)

    X_test = tvect.transform(test)
    print(classifier.predict_proba(X_test))
    print(classifier.predict(X_test))
