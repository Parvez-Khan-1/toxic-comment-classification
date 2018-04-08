from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import DataHelper
from src import Constants
from src.ExploratoryDataAnalysis import ExploratoryDataAnalysis
import nltk
import numpy as np
import time
from src import class_lookup

class CreateFeatureMatrix:

    def __init__(self):
        self.helper = DataHelper.DataHelper()

    def read_training_data(self):
        train_df = self.helper.read_csv(Constants.TRAIN_FILE)
        return train_df

    def read_test_data(self):
        test_df = self.helper.read_csv(Constants.TEST_FILE)
        return test_df

    def build_vocabulary(self, train_df):
        word_counter = set()
        all_comments = self.helper.get_values_in_specific_columns(train_df, 'comment_text')
        for comment in all_comments:
            word_counter.update(nltk.tokenize.word_tokenize(comment))
        return word_counter

    @staticmethod
    def build_feature_vector(train_df, vocab):
        feature_matrix = list()
        for row in train_df.iterrows():
            feature_matrix.append(CreateFeatureMatrix.get_word_frequencies(row, vocab))
        return feature_matrix

    @staticmethod
    def get_word_frequencies(row, vocab):
        comment_token = nltk.tokenize.word_tokenize(row[1]['comment_text'])
        word_freq = [row[1]['id']]
        for token in vocab:
            word_freq.append(comment_token.count(token))
        return word_freq


    @staticmethod
    def get_patterns(train_df):
        unique_patterns = list()
        for row in train_df.iterrows():
            toxic = row[1]['toxic']
            severe_toxic = row[1]['severe_toxic']
            obscene = row[1]['obscene']
            threat = row[1]['threat']
            insult = row[1]['insult']
            identity_hate = row[1]['identity_hate']
            sequence = [toxic, severe_toxic, obscene, threat, insult, identity_hate]
            if sequence not in unique_patterns:
                unique_patterns.append(sequence)

        print(len(unique_patterns))


    @staticmethod
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
            classes.append(list(class_lookup.pattern_lookup.keys())[list(class_lookup.pattern_lookup.values()).index(sequence)])
        print(len(classes))
        return classes


    @staticmethod
    def generate_tfidf_vector(data_frame):
        raw = data_frame.shape[0]
        print("train data set raw -", raw)
        # toxic_data_vector = TfidfVectorizer(ngram_range=(2, 3),
        #                                     min_df=3, max_df=0.5, strip_accents='unicode', use_idf=1,
        #                                     smooth_idf=1, sublinear_tf=1)
        toxic_data_vector = TfidfVectorizer(min_df=1, max_df=1)

        train_term_doc = toxic_data_vector.transform(data_frame['comment_text'])
        return train_term_doc

    @staticmethod
    def apply_logistic_regression(train_df, train_classes):

        toxic_data_vector = TfidfVectorizer(min_df=1, max_df=1)

        # toxic_data_vector = TfidfVectorizer(ngram_range=(2, 3),
        #                                     min_df=3, max_df=0.5, strip_accents='unicode', use_idf=1,
        #                                     smooth_idf=1, sublinear_tf=1)


        train_vec = toxic_data_vector.transform(train_df['comment_text'])

        classifier = LogisticRegression()
        classifier.fit(train_vec, train_classes)


        # toxic_data_vector = TfidfVectorizer(ngram_range=(2, 3),
        #                                     min_df=3, max_df=0.5, strip_accents='unicode', use_idf=1,
        #                                     smooth_idf=1, sublinear_tf=1)



        # test = ['Nonsense kiss off geek. what I said is true.  I ll have your account terminated.']
        # test_vec = toxic_data_vector.transform(test)
        # print(classifier.predict(test_vec))


class Test:

    if __name__ == '__main__':
        start_time = time.time()
        feature = CreateFeatureMatrix()

        train_df = feature.read_training_data()
        test_df = feature.read_test_data()[:100]
        filtered_train_df = DataHelper.DataHelper.filter_data_frame(train_df)

        #train_vec = feature.generate_tfidf_vector(filtered_train_df)
        train_classes = feature.get_pattern_code(filtered_train_df)

        #feature.apply_logistic_regression(filtered_train_df, train_classes)


        vocab = feature.build_vocabulary(filtered_train_df)

        ExploratoryDataAnalysis.show_class_wise_comment_count(filtered_train_df)

        # feature_matrix = feature.build_feature_vector(filtered_train_df, vocab)
        # print(feature_matrix)
        # narray = np.array(feature_matrix)
        # np.save('../data/feature_matrix', narray)
        # print(" Time Taken in Seconds : %s" % (time.time() - start_time))
