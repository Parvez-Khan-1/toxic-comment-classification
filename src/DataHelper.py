import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.Preprocessing import PreProcessing


class DataHelper:

    def read_csv(csv_path):
        try:
            return pd.read_csv(csv_path)
        except FileNotFoundError:
            print('Incorrect File Path')
            sys.exit(1)

    @staticmethod
    def get_values_in_specific_columns(data_frame, column_name):
        try:
            return data_frame[column_name].values
        except KeyError:
            print('Given Column Name Is Not Available In Data Frame')
            return None

    @staticmethod
    def data_frame_size(data_frame):
        return data_frame.shape[0]

    @staticmethod
    def filter_data_frame(data_frame):
        for index, row in enumerate(data_frame.iterrows()):
            # preprocessed_comment = PreProcessing.remove_stop_words(row[1]['comment_text'])
            lower_case_comment = PreProcessing.convert_to_lower_case(row[1]['comment_text'])
            special_char_filtered_comment = PreProcessing.removes_special_char(lower_case_comment)
            preprocessed_comment = PreProcessing.replace_multiple_spaces_with_single(special_char_filtered_comment)
            data_frame.set_value(index, 'comment_text', preprocessed_comment)
        return data_frame

    @staticmethod
    def create_sentences_for_gensim_word2vec(data_frame, interested_columns_name):
        sentences = []
        for row in data_frame.iterrows():
            comment_text = row[1]['comment_text']
            sentences.append(comment_text.split())
        return sentences

    @staticmethod
    def save_data_frame_to_csv(data_frame, csv_path):
        return data_frame.to_csv(csv_path, index=False)

    @staticmethod
    def split_data_set_to_train_test_using_scikit(data_frame):
        train, test = train_test_split(data_frame, test_size=0.3)
        return train, test

    def split_train_test_using_numpy(data_frame):
        df = data_frame
        msk = np.random.rand(len(df)) < 0.7
        train = df[msk]
        test = df[~msk]
        return train, test