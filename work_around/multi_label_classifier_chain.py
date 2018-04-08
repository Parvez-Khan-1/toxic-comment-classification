# using classifier chains
import sklearn
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset
from sklearn.naive_bayes import GaussianNB
from src.DataHelper import DataHelper
from src import class_lookup
import pandas as pd



def get_pattern_code(train_df):
    classes = []
    for row in train_df.iterrows():
        toxic = row[1]['toxic']
        severe_toxic = row[1]['severe_toxic']
        obscene = row[1]['obscene']
        threat = row[1]['threat']
        insult = row[1]['insult']
        identity_hate = row[1]['identity_hate']
        sequence = [toxic, severe_toxic, obscene, threat, insult, identity_hate]
        classes.append(
            list(class_lookup.pattern_lookup.keys())[list(class_lookup.pattern_lookup.values()).index(sequence)])
    return classes


def read_csv(file_path):
    return pd.read_csv(file_path)


# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
# classifier = ClassifierChain(GaussianNB())
# classifier = LinearSVC()
classifier = LogisticRegression()
# classifier = LabelPowerset(GaussianNB())

train_df = read_csv('../data/train.csv')
classes = get_pattern_code(train_df)
print(len(classes))

# corpus = ['Nonsense kiss off geek. what I said is true. I ll have your account terminated.',
#           'Please do not vandalize pages as you did with this edit to W. S. Merwin. If you continue to do so you will be blocked from editing',
#           'Why hasnt Alitalia been removed rom the alliance due to its piss-poor customer service - Sentiments of nummerous FlyerTalkers',
#           'Stupid As soon as I saw the phrase metal is stupid at the very beginning of the article, I assumed I misread it. When I confirmed that it did indeed say metal is stupid I then assumed it was vandalism. However, I dont know anything about chemistry See Soviet education system so if the word stupid has some other meaning in chemistry I would have no way of knowing or refuting it. Could someone that actually has some amount of knowledge on this subject confirm whether this is or is not vandalism or a typo or what have you',
#           'Ban one side of an argument by a bullshit nazi admin and you get no discussion because the islamist editors feel they won',
#           'Are you gay Sound like it',
#           'loser get a life you must be fat ugly and have no friends. that is why you are still up at 5 am in the morning when all the normal people are sleeping. get a life. that is pathetic that you spend all your life on here. and you are not even getting paid. what a loser. you sure dont have a life. yea. you are gonna be pissed. i know it. hahaha.']


toxic_data_vector = TfidfVectorizer(min_df=1, max_df=1)

X_train = toxic_data_vector.fit_transform(train_df['comment_text'])


# classes = ['1', '2', '1', '1', '3', '1', '5']
# Y_train = toxic_data_vector.fit_transform(classes)

print(classes)
# train
classifier.fit(X_train, classes)

test = ['you need a pussy your geek']
test_vec = toxic_data_vector.transform(test)
print(classifier.predict(test_vec))

# accuracy_score(y_test,predictions)