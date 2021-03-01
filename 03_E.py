import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")
dataset = pd.read_csv("IMDB Dataset.csv",)
dataset['tokenized_review'] = dataset['review'].apply(lambda x: nlp(x))
dataset
