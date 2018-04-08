import pandas as pd
import re
import nltk

data = pd.read_csv('../data/train.csv')


def get_toxic_data(dataset):
    return dataset.iloc[:, :3]


def get_divide_data(dataset):
    print('\ntotal data set is of', len(dataset), 'records')
    train_data = int(len(dataset) * 0.7)
    print('using training data of', train_data, 'records\n\n')
    data_is = get_toxic_data(data[:5])
    return data_is


def remove_special_characters(dataset):
    super_dict = dict()
    for line in dataset:
        clear_text = re.sub('\W+', ' ', line)
        pos_dict = get_tokenize(clear_text)
        super_dict = {**super_dict, **pos_dict}
    return super_dict

"""
https://www.webucator.com/how-to/how-merge-dictionaries-python.cfm

    grades = {**grades1, **grades2}

https://www.programiz.com/python-programming/methods/dictionary
https://www.digitalocean.com/community/tutorials/understanding-dictionaries-in-python-3

https://docs.python.org/3.3/tutorial/datastructures.html
"""


def get_tokenize(sentence):
    tokens = sentence.split()
    bag = dict(nltk.pos_tag(tokens))  # dict
    return bag



toxic_train_data = get_divide_data(data)
toxic_train_comments = toxic_train_data['comment_text']
toxic_train_comments = toxic_train_comments.str.lower()
pos_dict = remove_special_characters(toxic_train_comments)
print(pos_dict)