from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ExploratoryDataAnalysis:

    @staticmethod
    def visualize_word_embeddings(vocab):
        # train model
        sentences = [vocab]
        model = Word2Vec(sentences, min_count=1)
        # fit a 2d PCA model to the vectors
        X = model[model.wv.vocab]
        pca = PCA(n_components=2)
        result = pca.fit_transform(X)
        # create a scatter plot of the projection
        pyplot.scatter(result[:, 0], result[:, 1])

        words = list(model.wv.vocab)
        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()

    @staticmethod
    def show_class_wise_comment_count(data_frame):
        countdata = data_frame.iloc[:, 2:].apply(pd.Series.value_counts)
        print(countdata)
        n_groups = 6

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, countdata.iloc[0,], bar_width,
                         alpha=opacity,
                         color='b',
                         label='1')

        rects2 = plt.bar(index + bar_width, countdata.iloc[1,], bar_width,
                         alpha=opacity,
                         color='g',
                         label='0')

        plt.xlabel('Toxic Comment')
        plt.ylabel('Count')
        plt.title('Scores by Toxic Comment')
        plt.xticks(index, countdata.columns.values)
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_class_wise_count(data_frame):
        x = data_frame.iloc[:, 2:].sum()
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

        for idx, val in enumerate(x.values):
            labels[idx] = str(labels[idx]+"_"+str(val))
        # this is for plotting purpose
        index = np.arange(len(labels))
        plt.bar(index, x, 0.50)
        plt.xlabel('Comment Class', fontsize=10)
        plt.ylabel('No of Comments', fontsize=10)
        plt.xticks(index, labels, fontsize=10, rotation = 0.30)
        plt.title('Class Wise Count')
        plt.show()

    @staticmethod
    def visualize_word_vec(vector, model):
        pyplot.scatter(vector[:, 0], vector[:, 1])
        words = list(model.wv.vocab)
        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(vector[i, 0], vector[i, 1]))
        pyplot.show()