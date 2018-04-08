from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

corpus = [
    "I am super good with Java and JEE",
    "I am super good with .NET and C#",
    "I am really good with Python and R",
    "I am really good with C++ and pointers"
    ]

classes = ["java developer", ".net developer", "data scientist", "C++ developer"]

test = ["I think I'm a good developer with really good understanding of .NET"]

tvect = TfidfVectorizer(min_df=1, max_df=1)

X = tvect.fit_transform(corpus)

classifier = LinearSVC()
classifier.fit(X, classes)


X_test=tvect.transform(test)
print(classifier.predict(X_test))