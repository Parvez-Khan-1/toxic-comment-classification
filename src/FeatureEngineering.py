from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from src import DataHelper


def initialized_count_vectorizer(data_frame):
    vectorizer = CountVectorizer()
    vectorizer.fit(data_frame.comment_text)
    return vectorizer


def apply_count_vectorizer(vectorizer, data_frame):
    vec = vectorizer.transform(data_frame.comment_text)
    return vec


def initialized_tf_idf_vectorizer(data_frame):
    vectorizer = TfidfVectorizer(lowercase=False)
    vectorizer.fit(data_frame.comment_text)
    return vectorizer


def apply_tf_idf_vectorizer(vectorizer, data_frame):
    vec = vectorizer.transform(data_frame.comment_text)
    return vec


def convert_word2vec(train_df):
    comment_text = DataHelper.DataHelper.create_sentences_for_gensim_word2vec(train_df, 'comment_text')
    model = Word2Vec(comment_text, min_count=1)
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # data_exploration.ExploratoryDataAnalysis.visualize_word_vec(result, model)
    return result