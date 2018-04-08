from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# load the Stanford GloVe model
glove_input_file = '/home/synerzip/Desktop/Work/neuro-ner/neuro-ner/data/word_vectors/glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

# load the Stanford GloVe model
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)


# calculate: (king - man) + woman = queen
# calculate: (man - woman) + queen = king

result = model.most_similar(positive=['man', 'queen'], negative=['woman'], topn=1)
print(result)