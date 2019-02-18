# Twitter-data-analysis
Twitter data analysis using Bag of Words methodolgy 

The main objective is mainly to predict the polarity of tweets provided in the dataset , the dataset for the sake of simplicity and testing the accuracy of proposed model ,contains a column containing polarity of the respective tweets.

PREPROCESSING

The first step involves preprocessing of the tweets.Preprocessing of the tweets is an essential step as it makes the raw text ready for mining. 1) The Twitter handles marked by usually '@' hardly comnvey any useful info hence are removed. 2) Getting rid of punctuation marks. 3) Most of the smaller do not add much value (eg. 'in','all','edx'). Remove them too.


TOKENIZATION

Tokens are individual words ot terms, and tokenization is the process of splitting a string of text into tokens, easily done using split() function.

STEMMING

Stemming is a rule-based process of stripping words of their suffixes('ing', 'ly','es','s') making several words a simple variation of the same word.This part is executed using NLTK library of python.


VISUALIZTION

WordClouds is a visualization wherein the most frequent words appear in large size and the less frequent  words appear in smaller sizes. It also us to understand the correctness of the polarity  of words.

FEATURE EXTRACTION 

    BAG-OF-WORDS
    Bag of Words is a method to represent text into numerical features. The bag-of-words model is a simplifying representation used in natural language processing and information retrieval (IR). In this model, a text (such as a sentence or a document) is represented as the bag (multiset) of its words, disregarding grammar and even word order but keeping multiplicity.(https://en.wikipedia.org/wiki/Bag-of-words_model)
    Bag of words can be easily be created using CountVectorizer library .(documentation:- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html#examples-using-sklearn-feature-extraction-text-countvectorizer
    Usage:-https://adataanalyst.com/scikit-learn/countvectorizer-sklearn-example/)
    
 MODEL BUILDING
 
 I used  the scikit learn logistic regression classifier to build the multi-classification classifier and then have used the metrics library to compute the accuracy our model against the polarity already provided.
 (http://dataaspirant.com/2017/05/15/implement-multinomial-logistic-regression-python/)
 
    

