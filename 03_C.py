import numpy as np
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import nltk

class NaiveBayes:
    def __init__(self, smoothing=False):
        self.smoothing = smoothing
        self.weights = None
        self.word2idx = {}
        self.positive_prob = 0
        self.negative_prob = 0
    
    def fit(self, X, y):
        self.fill_word2idx(X)
        self.positive_prob = np.sum(y == 1) / y.shape[0]
        self.negative_prob = 1 - self.positive_prob
        self.weights = np.zeros((len(y.unique()), len(self.word2idx)))

        for i, review in enumerate(X):
            review = review.strip().replace('<br />', '') 
            for word in review.split():
                if word not in self.word2idx: continue
                self.weights[y[i], self.word2idx[word]] += 1

        if self.smoothing:
            self.weights = ((self.weights + 1) / 
                            (np.sum(self.weights, axis=1).reshape(-1, 1) + len(self.word2idx)))
        else:
            self.weights = (self.weights) / (np.sum(self.weights, axis=1).reshape(-1, 1))

    def predict(self, X):
        y_pred = np.zeros_like(X)
        for i, review in enumerate(X):
            review = review.strip().replace('<br />', '')
            positive_sum = np.log(self.positive_prob)
            negative_sum = np.log(self.negative_prob)
            for word in review.split():
                if word not in self.word2idx: continue
                positive_sum += np.log(self.weights[1, self.word2idx[word]])
                negative_sum += np.log(self.weights[0, self.word2idx[word]])
            if positive_sum > negative_sum:
                y_pred[i] = 1
        return y_pred
                
    def fill_word2idx(self, X):
        idx = 0
        for review in X:
            review = review.strip().replace('<br />', '')
            for word in review.split():
                if (word not in punctuation and 
                        word not in stopwords.words('english') and 
                        word not in self.word2idx):
                    self.word2idx[word] = idx
                    idx += 1
df = pd.read_csv('C:/Users/fatkh/OneDrive/3 course 2 term/Natural Language Processing/assignmnet 3/IMDB Dataset.csv')
X, y = df['review'], df['sentiment'].map({'positive': 1, 'negative': 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

clf = NaiveBayes(smoothing=True)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred = pd.Series(y_pred).astype(np.int64)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n"
      f"Precision: {precision_score(y_test, y_pred)}\n"
      f"Recall: {recall_score(y_test, y_pred)}\n"
      f"F1: {f1_score(y_test, y_pred)}")
