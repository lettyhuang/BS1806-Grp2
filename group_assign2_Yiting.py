# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:30:56 2017

@author: Yiting
"""

import pandas as pd
import numpy as np
import nltk as nl
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.utils import shuffle

#load data into dataframe
df = pd.read_table('C:/SMSSpamCollection', header=None, names=["classifier","text"])

"""
Pre-process the SMS messages: Remove all punctuation and numbers from the SMS
messages, and change all messages to lower case. (Please provide the Python code that
achieves this!)
"""

#remove numbers
df['text'] = df['text'].apply(lambda x: re.sub("\d+", " ", x))

#sentence lower case
df['text'] = df['text'].apply(lambda x: x.lower())

#remove punctuation 
tokenizer = RegexpTokenizer(r'\w+')
df['text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))
df['text'] = df['text'].apply(lambda x: ' '.join(x))

#above credit to letty!

#Shuffle the messages
df = shuffle(df)

#split df
df_train = df[:2500]  #2500 messages
df_validation = df[2500:1000] #1000 messages
df_test = df[3500:] #2072 messages remaining

# explain
class NaiveBayesForSpam:
    def train (self, hamMessages, spamMessages):
        self.words = set (' '.join (hamMessages + spamMessages).split())
        self.priors = np.zeros(2)
        self.priors[0] = float (len (hamMessages)) / (len (hamMessages) + len(spamMessages))
        self.priors[1] = 1.0-self.priors[0] 
        self.likelihoods = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len(hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len(spamMessages)
            self.likelihoods.append([min(prob1, 0.95), min(prob2, 0.95)])
        self.likelihoods = np.array (self.likelihoods).T
    """
    function train
    input: a list of ham messages, a list of spam messages
    attribute? methods?:
    -words: a set of unique words; get all messages (ham&spam) from training data, split them into words
    -priors: a np array of prior probabilities; show the prior probabilities of a message being ‘ham’ (priors[0]) or‘spam’ (priors[1])
    -likelihoods: a np array of likelyhoods -- array([[P(word|ham)], [P(word|spam]]); 
                  for each unique word in the training set (elements in "words"), cacluate the likelyhood 
                  prob1 = P(word|ham), prob2 = P(word|spam); Laplace estimator - add 1 to numerator of prob1 and prob 2 to avoid zero; 
                  replace any likelihood larger than 0.95 with 0.95
    """
    def train2 (self, hamMessages, spamMessages) :
        self.words = set (' '.join (hamMessages + spamMessages).split()) 
        self.priors = np.zeros(2)
        self.priors[0] = float (len (hamMessages)) / (len (hamMessages) + len(spamMessages))
        self.priors[1] = 1.0 - self.priors[0] 
        self.likelihoods = []
        spamkeywords = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) /len(hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len(spamMessages) 
            if prob1 * 20 < prob2:
                self.likelihoods.append([min(prob1, 0.95), min(prob2, 0.95)])
                spamkeywords.append (w) 
        self.words = spamkeywords
        self.likelihoods = np.array (self.likelihoods).T
    """
    function train2
    input: ham messages, spam messages
    attribute? methods?:
    -words: a set of spamkeywords; for a word, if P(word|ham) * 20 < P(word|spam), it is classified as spamkeywords and added to words
    -priors: a np array of prior probabilities; show the prior probabilities of a message being ‘ham’ (priors[0]) or‘spam’ (priors[1])
    -likelihoods: for a word, if P(word|ham) * 20 < P(word|spam), which means this word appears quite more frequently in spam than
                 in ham messages, it appears in the np array of likelyhoods
    By add the judgement condition, the function record the most important spam keywords and their likelyhoods; 
    save workload and increase speed?
    """
    def predict(self, message) :
        posteriors = np.copy(self.priors)
        for i, w in enumerate(self.words):
            if w in message.lower(): # convert to lower−case
                posteriors *= self.likelihoods[:,i] 
            else :
                posteriors *= np.ones (2)-self.likelihoods[:,i] 
            posteriors = posteriors / np.linalg.norm (posteriors , ord =1) # normalise 
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]] 
        return['spam', posteriors[1]]
    """
    Bayes' Theorem being applied in the above code
    input: a new message 
    posteriors: a np array of posterior probabilities; P(ham|message) = posteriors[0], P(spam|message) = posteriors[1]
    output: whether the new message is ham or spam, and the posterior probability
    """
    def score (self, messages, labels): 
        confusion = np.zeros(4).reshape(2 ,2) 
        for m, l in zip(messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion [0,0] += 1 
            elif self.predict(m)[0] == 'ham' and l == 'spam':                 
                confusion [0,1] += 1 
            elif self.predict(m)[0]== 'spam' and l == 'ham':
                confusion [1,0] += 1 
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion [1,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum()) , confusion
    """
    input: messages and labels (ham or spam)
    output: the accuracy rate of Naive Bayes model, and confusion matrix
    something confusing: is spam or ham the important one in confusion matrix?
    confusion matrix looks strange...
    """

# Use training set to train the classifiers ‘train’ and ‘train2’

hamMessages = list(df_train[df_train.classifier == "ham"]["text"])
spamMessages = list(df_train[df_train.classifier == "spam"]["text"])

m1 = NaiveBayesForSpam()
m1.train(hamMessages, spamMessages)

m2 = NaiveBayesForSpam()
m2.train2(hamMessages, spamMessages)


# Using the validation set, explore how each of the two classifiers performs out of sample.
messages =  list(df_validation["text"])
labels = list(df_validation["classifier"])
m1.score(messages, labels)
m2.score(messages, labels)

#  Why is the ‘train2’ classifier faster? 
# Why does it yield a better accuracy both on the training and the validation set
"""
train2 do some filtering on spamkeywords, increase the importance of spam keywords
"""
# How many false positives (ham messages classified as spam messages) did you get in your validation set? 
# How would you change the code to reduce false positives at the expense of possibly having more false negatives (spam messages classified as ham messages)?

FP1 = 25
FP2 = 2 # not sure about this question

# Run the ‘train2’ classifier on the test set and report its performance using a confusion matrix.
messages_test = list(df_test["text"])
labels_test = list(df_test["classifier"])
m2.score(messages_test, labels_test)
