#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 14:55:26 2019

@author: ashutosh
"""
# Importing required libraries
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import string
import nltk

# Reading dataset
dataset= pd.read_csv('twitter_data.csv')
#function to remove @ handles from tweets as they are useless
def remove_pattern(input_txt, pattern):
    r=re.findall(pattern, input_txt)
    for i in r:
        input_txt=re.sub(i, '', input_txt)
    return input_txt
#removing @ twitter handles
dataset['tidy_tweet']=np.vectorize(remove_pattern)(dataset['tweet'],"@[\w]*")
#removing words shorter than 3 as they generally dont have impact on meaning

dataset['tidy_tweet']=dataset['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
#tokenizing the cleaned dataframe
tokens=dataset['tidy_tweet'].apply(lambda x: x.split())

from nltk.stem.porter import *
# Stemmer to remove suffixes like:'ing','-ly' etc that donot much influence the meaning
stemmer = PorterStemmer()
tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])

tokens.head()

for i in range(len(tokens)):
    tokens[i]=','.join(tokens[i])
    '''

i=0
for i in range(len(tokens)):
    #print(np.array(tokens[i]))
    dataset.at[i, 'tidy_tweet']=np.array(tokens[i])
    i=i+1
'''
#restoring the cleaned tweets to dataset
dataset['tidy_tweet']=tokens

# way to display wordcloud with font proportion to count 
words = ' '.join([text for text in dataset['tidy_tweet']])
from wordcloud import WordCloud
wordcloud=WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(words)
 
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#neutral words polarity == 2
normal_words = ' '.join([text for text in dataset['tidy_tweet'][dataset['polarity']==2]])

wordcloud=WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
 
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#negative words polarity==4
neg_words=' '.join([text for text in dataset['tidy_tweet'][dataset['polarity']==0]])

wordcloud=WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(neg_words)
 
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
'''
def hashtag_extract(x):
    hashtags=[]
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags

hash_neg = hashtag_extract(dataset['tidy_tweet'][dataset['polarity']==0])
hash_pos = hashtag_extract(dataset['tidy_tweet'][dataset['polarity']==4])
'''
#importing CountVectorizer that does counting and tokenizing both
#we are using the BAG-OF-WORDS method to build classifier
from sklearn.feature_extraction.text import CountVectorizer
bag = CountVectorizer(stop_words='english', max_features=1000, lowercase = False)
bagofwords = bag.fit_transform(dataset['tidy_tweet'])


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


xtrain_bag, xvalid_bag, ytrain, ytrain_bag = train_test_split(bagofwords, dataset['polarity'], random_state=42, test_size=0.3)
from sklearn import linear_model
from sklearn import metrics
#multinomial logistic Regression using 'newton-cg
mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(xtrain_bag, ytrain)

#accuracy testing for train set
accuracy_train = metrics.accuracy_score(ytrain, mul_lr.predict(xtrain_bag))

#accuracy testing for train set
accuracy_test = metrics.accuracy_score(ytrain_bag, mul_lr.predict(xvalid_bag))
#can test using visual tools if needed
test_pred = mul_lr.predict(xvalid_bag)


