from flask import Flask ,request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from collections import Counter
from wordcloud import WordCloud
from os import path

class wordtoplot:
    def __init__(self,doc):
        self.doc=doc
    def clean_text(self):
        text=re.sub('[^a-zA-z]',' ',str(self.doc))
        text=re.sub('\[.*?\]',' ',text)
        text=re.sub('\d',' ',text)
        text=" ".join(text.split())
        text=text.lower()
        return text
        
    def do_lemma(self):
        # print(self.clean_text())
        sentences = nltk.sent_tokenize(self.clean_text())
        words = nltk.word_tokenize(self.clean_text())
        lemmatizer=WordNetLemmatizer()
        for i in range(len(sentences)):
            words=nltk.word_tokenize(sentences[i])
            words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
            sentences[i] = ' '.join(words) 
        return sentences


    def make_df(self):
        lst=['like','im','know','just','dont','house','people','thats','right',\
    'got' ,'time','you' ,'said','yeah','okay','say','transcript','want','say','let','oh','nice','hey',\
    'coming','come','make','thing','need']
        doc_df=pd.DataFrame(self.do_lemma(),columns=['transcript'],index=['doc'])
        stop_words=[word for word ,counts in Counter(lst).most_common() if counts>=1]
        stop_words=text.ENGLISH_STOP_WORDS.union(stop_words)
        cv=CountVectorizer(stop_words='english')
        data_cv=cv.fit_transform(doc_df.transcript)
        data_dtm=pd.DataFrame(data_cv.toarray(),columns=cv.get_feature_names())
        data_dtm.index=doc_df.index
        data_dtm.to_pickle('dtm.pkl')
        return stop_words,doc_df

    def word_plot(self):
        d="E:/Notebook-file/NLP.idea/static/images"
        wc = WordCloud(stopwords=self.make_df()[0], background_color="white", colormap="Dark2",
                max_font_size=150, random_state=42)
        plt.rcParams['figure.figsize'] = [15, 6]

        cloud=wc.generate(self.make_df()[1].transcript['doc'])
        # plt.title("More Often Words With Result")
        return cloud.to_file(path.join(d,'cloud.png')
       


