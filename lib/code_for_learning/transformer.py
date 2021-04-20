import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

nltk.download("stopwords")


class Vectorizer(object):
    def __init__(self, text_col, cat_cols):
        self.ohe = OneHotEncoder()
        self.tfidf = TfidfVectorizer(
            min_df=10,
            max_df=0.9,
            stop_words=stopwords.words("russian"),
            ngram_range=(1, 3),
            max_features=30000
        )
        self.text_col = text_col
        self.cat_cols = cat_cols
        self.columns_ = None

    def fit(self, X):
        self.tfidf = self.tfidf.fit(X[self.text_col])
        self.ohe = self.ohe.fit(X[self.cat_cols])
        self.columns_ = list(self.tfidf.get_feature_names()) + list(self.ohe.get_feature_names()) + ["price"]
        return self

    def transform(self, X):
        # tfidf_features = self.tfidf.transform(X[self.text_col])
        # cat_features = self.ohe.transform(X[self.cat_cols])
        x = X.loc[:, ["price"]]
        x["price"] = np.log10(x["price"].fillna(0.0) + 1.0)
        return hstack((self.tfidf.transform(X[self.text_col]), self.ohe.transform(X[self.cat_cols]), x[["price"]]))

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
