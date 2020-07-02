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
from PIL import Image
from os import path
from nltk import word_tokenize, pos_tag
from gensim import matutils, models
import scipy.sparse


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
        d="E:\\Notebook-file\\NLP.idea\\static\\images"
        lst=['like','im','know','just','dont','house','people','thats','right',\
    'got' ,'time','you' ,'said','yeah','okay','say','transcript','want','say','let','oh','nice','hey',\
    'coming','come','make','thing','need']
        doc_df=pd.DataFrame(self.do_lemma(),columns=['transcript'],index=['doc'])
        doc_df.to_pickle('dtm_clean.pkl')
        stop_words=[word for word ,counts in Counter(lst).most_common() if counts>=1]
        stop_words=text.ENGLISH_STOP_WORDS.union(stop_words)
        cv=CountVectorizer(stop_words='english')
        data_cv=cv.fit_transform(doc_df.transcript)
        data_dtm=pd.DataFrame(data_cv.toarray(),columns=cv.get_feature_names())
        data_dtm.index=doc_df.index
        data_dtm.to_pickle('dtm.pkl')
        return stop_words,doc_df

    def word_plot(self):
        d="E:\\Notebook-file\\NLP.idea\\static\\images"
        wc = WordCloud(stopwords=self.make_df()[0], background_color="white", colormap="Dark2",
                max_font_size=150, random_state=42)
        plt.rcParams['figure.figsize'] = [13, 5]

        cloud=wc.generate(self.make_df()[1].transcript['doc'])
        # plt.title("More Often Words With Result")
        cloud.to_file(path.join(d,'cloud.png'))

    def common_plot(self):
        d="E:\\Notebook-file\\NLP.idea\\static\\images"
        df= pd.read_pickle("E:/Notebook-file/NLP.idea/dtm.pkl")
        df=df.transpose()
        df.sort_values(by='doc',ascending=False,inplace=True)
        top_20=df[:20]
        height = sorted(top_20['doc'].values)
        bars = tuple(top_20.index)[::-1]
        y_pos = np.arange(len(bars))
        plt.rcParams["figure.figsize"]=[15,6]
        plt.barh(y_pos, height)
        plt.yticks(y_pos, bars)
        plt.savefig(path.join(d,'common.png'))

    def nouns(self,text):
        '''Given a string of text, tokenize the text and pull out only the nouns.'''
        is_noun = lambda pos: pos[:2] == 'NN'
        tokenized = word_tokenize(text)
        all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)] 
        return ' '.join(all_nouns)

    def make_df2(self):
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
    
    def get_topics(self):
        lst_topics=list()
        corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(self.make_df2()[0].transpose()))
        id2word = dict((v, k) for k, v in self.make_df2()[1].vocabulary_.items())
        lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=80)
        ans=lda.print_topics()
        corpus_transformed = lda[corpus]
        lst_topic=list(zip([a for [(a,b)] in corpus_transformed], self.make_df2()[0].index))
        fall=lst_topic[0][0]
        topics=ans[fall][1]
        x=re.sub('[^a-zA-z]',' ',topics)
        topics=" ".join(x.split())
        for i in topics.split():
            lst_topics.append(i)
        return lst_topics