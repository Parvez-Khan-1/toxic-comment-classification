import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier # import random forest classifier from sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # import two tokenizers from sklearn
# Input data files are available in the "../input/" directory.


import os
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
subm = pd.read_csv('../data/sample_submission.csv')

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)



train['comment_text'].fillna("unknown", inplace=True)
test['comment_text'].fillna("unknown", inplace=True) # filling in NaN comments with "unknown"

import re, string
re_tok = re.compile(r'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

# n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )

train_term_doc = vec.fit_transform(train['comment_text'])
test_term_doc = vec.transform(test['comment_text'])

preds = np.zeros((len(test), len(label_cols))) # empty np matrix to put in predictions

for i, j in enumerate(label_cols):
    print('fit', j)
    m = RandomForestClassifier(n_estimators=1000, max_leaf_nodes=18, random_state=21)
    m.fit(train_term_doc, train[j].values)
    preds[:,i] = m.predict_proba(test_term_doc)[:,1]

submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission_me.csv', index=False)


