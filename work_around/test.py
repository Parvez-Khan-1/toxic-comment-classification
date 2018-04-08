import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('../data/train.csv')[:20000]
test = pd.read_csv('../data/test.csv')[:20000]
print('load data')

vec = TfidfVectorizer()
# vec = CountVectorizer()
# vec.fit(pd.concat([train.comment_text, test.comment_text]))
vec.fit(train.comment_text)

x_train = vec.transform(train.comment_text)
x_test = vec.transform(test.comment_text)
print('prepare data')
print(x_train.shape)
print(x_test.shape)

sub = pd.DataFrame()
sub['id'] = test['id']

targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
for t in targets:
    c = LogisticRegression(C=1.2, class_weight='balanced')
    c.fit(x_train, train[t].values)
    y_pred = c.predict_proba(x_test)
    idx = list(c.classes_).index(1)
    sub[t] = y_pred[:, idx]
    print('predict {}'.format(t))
sub.to_csv('../data/submission.csv', index=False)







