import numpy as np
from scipy.optimize import minimize
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pymorphy3
import cirq
import scipy

# векторизация
vectorizer = TfidfVectorizer(max_features=128)
banned = stopwords.words("russian") + [",", ".", ";", ":", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
morph = pymorphy3.MorphAnalyzer()


def csv2vec(data):
    t_data = [word_tokenize(i) for i in data]  # tokenized data
    l_data = [" ".join(
        [morph.parse(t_data[i][j])[0].normal_form for j in range(len(t_data[i])) if t_data[i][j] not in banned]) for i
        in range(len(t_data))]  # lemmatized data
    matrix = vectorizer.fit_transform(pd.DataFrame(l_data)[0]).toarray()
    return matrix


data = pd.read_csv("task-3-dataset.csv")
reviews0 = data["отзывы"].to_list()
review_class0 = data["разметка"].to_list()
reviews = reviews0[:60]
review_class = review_class0[:60]
func = lambda x: 1 if x == "+" else 0
review_class = np.array(list(map(func, review_class)))
matrix = csv2vec(reviews)