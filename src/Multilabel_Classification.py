from src import DataHelper
from src import Constants
from src import FeatureEngineering
from src.ExploratoryDataAnalysis import ExploratoryDataAnalysis as EDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from src import Evaluation


def do_preprocessing(data_frame):
    return DataHelper.DataHelper.filter_data_frame(data_frame)


def do_model_evaluation(predicted_result, actual_result):
    evaluation_results = dict()
    evaluation_results["accuracy"] = Evaluation.get_model_accuracy(actual_result, predicted_result)
    evaluation_results["confusion_matrix"] = Evaluation.show_confusion_matrix(actual_result, predicted_result)
    return evaluation_results


def show_evaluation(evaluation_dict):
    for key, value in evaluation_dict.items():
        print(key)
        print(value)
    print('-' * 20)


def train_classifier_using_logistic_regression(train_vectorizer, train_df, test_df, test_vectorizer):
    final_results = pd.DataFrame()
    final_results['id'] = test_df['id']
    for each_class in Constants.TARGET_CLASSES:
        algo = LogisticRegression()
        algo.fit(train_vectorizer, train_df[each_class].values)
        prediction = algo.predict_proba(test_vectorizer)
        final_results[each_class] = prediction[:, 1]
    final_results.to_csv(Constants.RESULT_FILE, index=False)


def train_classifier_using_multinaive_bayes(train_vectorizer, train_df, test_df, test_vectorizer):
    final_results = pd.DataFrame()
    final_results['id'] = test_df['id']
    for each_class in Constants.TARGET_CLASSES:
        algo = LogisticRegression()
        algo.fit(train_vectorizer, train_df[each_class].values)
        prediction = algo.predict_proba(test_vectorizer)
        final_results[each_class] = prediction[:, 1]
    final_results.to_csv(Constants.RESULT_FILE, index=False)


if __name__ == '__main__':

    train_data = DataHelper.DataHelper.read_csv(Constants.TRAIN_FILE)
    test_data = DataHelper.DataHelper.read_csv(Constants.TEST_FILE)
    print('train and test set loaded Successfully..')

    # EDA.show_class_wise_count(train_data)
    # print('Exploratory Data Analysis Done..')

    train_data = do_preprocessing(train_data)
    test_data = do_preprocessing(test_data)
    print("Preprocessing Done Successfully...")

    vectorizer = FeatureEngineering.initialized_tf_idf_vectorizer(train_data)
    print('Vectorizer Intialized...')

    train_vectorizer = FeatureEngineering.apply_tf_idf_vectorizer(vectorizer, train_data)
    test_vectorizer = FeatureEngineering.apply_tf_idf_vectorizer(vectorizer, test_data)
    print('Feature Extraction Completed')

    train_classifier_using_logistic_regression(train_vectorizer, train_data, test_data, test_vectorizer)
    print("Finished, Results store in Output/submission.csv")

