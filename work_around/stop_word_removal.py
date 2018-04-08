from nltk import word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
sentence = "this is a foo bar sentence"
print([i for i in sentence.lower().split() if i not in stop])

print(' '.join([i for i in word_tokenize(sentence.lower()) if i not in stop]))
