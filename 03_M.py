import spacy
import pandas as pd
import numpy as np
from string import punctuation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

df = pd.read_csv('C:/Users/fatkh/OneDrive/3 course 2 term/Natural Language Processing/assignmnet 3/IMDB Dataset.csv')
nlp = spacy.load('en_core_web_sm')
df['tokens'] = df['review'].apply(lambda doc: [token.text for token in nlp(doc) if token.text not in punctuation])

X = df['tokens']
y = df['sentiment']

def dictionary(file_path='C:/Users/fatkh/OneDrive/3 course 2 term/Natural Language Processing/assignmnet 3/2000.tsv'):
    dictionary = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            word, mean_sentiment, _ = line.split('\t')
            dictionary[word] = float(mean_sentiment)
    return dictionary
    
 def cms(tokens, dictionary):
    sentiment_sum = 0
    for token in tokens:
        if token in dictionary:
            sentiment_sum += dictionary[token]
    return sentiment_sum
   
 dictionary = dictionary()
 y_pred = df['tokens'].apply(lambda x: 'positive' if cms(x, dictionary) >= 0 else 'negative')
 
 print(f"Accuracy: {accuracy_score(y, y_pred)}\n"
      f"Precision: {precision_score(y, y_pred, pos_label='positive')}\n"
      f"Recall: {recall_score(y, y_pred, pos_label='positive')}\n"
      f"F1: {f1_score(y, y_pred, pos_label='positive')}")
