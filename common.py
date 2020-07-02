from flask import Flask
from flask_classful import FlaskView
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from gensim import matutils, models
import scipy.sparse

import pickle

class TestView():
    def nouns(self,text):
        '''Given a string of text, tokenize the text and pull out only the nouns.'''
        is_noun = lambda pos: pos[:2] == 'NN'
        tokenized = word_tokenize(text)
        all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
        return ' '.join(all_nouns)

    def index(self):
    # http://localhost:5000/
        df=pd.read_pickle('dtm_clean.pkl')
        data_nouns = pd.DataFrame(df.transcript.apply(self.nouns))
        stop_words=['like','im','know','just','dont','house','people','thats','right',\
         'got' ,'time','you' ,'said','yeah','okay','say','transcript','want','say','let','oh','nice','hey',\
        'coming','come','make','thing','need']

        stop_words=text.ENGLISH_STOP_WORDS.union(stop_words)
        cv=CountVectorizer(stop_words=stop_words)
        data_cv=cv.fit_transform(data_nouns.transcript)
        data_stop=pd.DataFrame(data_cv.toarray(),columns=cv.get_feature_names())
        data_stop.index=df.index
        return data_stop,cv
    
    def get(self):
        corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(self.index()[0].transpose()))
        id2word = dict((v, k) for k, v in self.index()[1].vocabulary_.items())
        lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=80)
        ans=lda.print_topics()
        corpus_transformed = lda[corpus]
        lst_topic=list(zip([a for [(a,b)] in corpus_transformed], self.index()[0].index))
        fall=lst_topic[0][0]
        topics=ans[fall][1]
        f = open("myfile.txt", "w")
        f.write(topics)
        return topics

obj= TestView()
obj.get()