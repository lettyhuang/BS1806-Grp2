# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:24:02 2017

@author: letty
"""

import pandas as pd
import numpy as np
from numpy import linalg 
import re
from nltk.tokenize import RegexpTokenizer

#load data into dataframe
df = pd.read_table('SMSSpamCollection.txt',header=None, names=["classifier","text"])


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

"""
Shuffle the messages and split them into a training set (2,500 messages), a validation
set (1,000 messages) and a test set (all remaining messages).
"""
df=df.sample(frac=1).reset_index(drop=True)

dfTrain=df.loc[0:2499,:]
dfValidation=df.loc[2500:3499,:]
dfTest=df.loc[3500:len(df),:]

"""
Explain the code: What is the purpose of each function? What do 'train' and `train2'
do, and what is the diffence between them? Where in the code is Bayes' Theorem
being applied?
"""
class NaiveBayesForSpam:
    def train (self, hamMessages, spamMessages):
        self.words = set (' '.join (hamMessages + spamMessages).split())
        self.priors = np.zeros (2)
        self.priors[0] = float (len (hamMessages)) / (len (hamMessages) + len (spamMessages))
        self.priors[1] = 1.0 - self.priors[0]
        self.likelihoods = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len (hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len (spamMessages)
            self.likelihoods.append ([min (prob1, 0.95), min (prob2, 0.95)])
        self.likelihoods = np.array (self.likelihoods).T
        
        """
        The purpose of the above "train" function is to first calculate the prior probability,
        and then create a likelihood table for each word (P(word|Spam) & P(word|Ham)
        in order to further be used to calculate 
        the posterior probability and determine whether they are more likely to be spam or ham 
        using Naive Bayes Algorithm. Notice that each likehood will not exceed 0.95 since the 
        function restricts the highest likelihood to 0.95.
        """
        
    def train2 (self, hamMessages, spamMessages):
        self.words = set (' '.join (hamMessages + spamMessages).split())
        self.priors = np.zeros (2)
        self.priors[0] = float (len (hamMessages)) / (len (hamMessages) + len (spamMessages))
        self.priors[1] = 1.0 - self.priors[0]
        self.likelihoods = []
        spamkeywords = []
        for i, w in enumerate (self.words):
            prob1 = (1.0 + len ([m for m in hamMessages if w in m])) / len (hamMessages)
            prob2 = (1.0 + len ([m for m in spamMessages if w in m])) / len (spamMessages)
            if prob1 * 20 < prob2:
                self.likelihoods.append ([min (prob1, 0.95), min (prob2, 0.95)])
                spamkeywords.append (w)
        self.words = spamkeywords
        self.likelihoods = np.array (self.likelihoods).T
        
        
        """
        The purpose of the above "train2" function is to first calculate the prior probability
        of ham and spam message given the number of messages in each category. Next, the function 
        will return a likelihood table just as what "train" function does. However, "train2" 
        function will compare each word's likelihood in spam/ham message and if the Spam prob is 20
        times higher the Ham prob, it means that the word has a very high appreance in Spam and
        thus the function will create a list of these words and classified as Spam key word.
        """

    def predict (self, message):
        posteriors = np.copy (self.priors)
        for i, w in enumerate (self.words):
            if w in message.lower():  # convert to lower-case
                posteriors *= self.likelihoods[:,i]
            else:                                   
                posteriors *= np.ones (2) - self.likelihoods[:,i]
            posteriors = posteriors / linalg.norm (posteriors)  # normalise
        if posteriors[0] > 0.5:
            return ['ham', posteriors[0]]
        return ['spam', posteriors[1]]    
    
    """
    The purpose of the above "predict" function is predict whether a new input message
    is ham or spam. This function first identify whether each word within the input message
    is in the spam key word list or not, then further calculate the posterior probability using
    Naive Bayes Algorithm. This is whether Bayes' Theorem is applied.
    """

    def score (self, messages, labels):
        confusion = np.zeros(4).reshape (2,2)
        for m, l in zip (messages, labels):
            if self.predict(m)[0] == 'ham' and l == 'ham':
                confusion[0,0] += 1
            elif self.predict(m)[0] == 'ham' and l == 'spam':
                confusion[0,1] += 1
            elif self.predict(m)[0] == 'spam' and l == 'ham':
                confusion[1,0] += 1
            elif self.predict(m)[0] == 'spam' and l == 'spam':
                confusion[1,1] += 1
        return (confusion[0,0] + confusion[1,1]) / float (confusion.sum()), confusion
    
    """
    The purpose of the above "score" function is to return a confusion matrix specifying
    the overall predict result from the "predict" function. We can see from the matrix 
    whether the Naive Bayes Classification return the correct classifier of the input message.
    """


"""
Use your training set to train the classifiers `train' and `train2'. Note that the interfaces
of our classifiers require you to pass the ham and spam messages separately.
"""
#pass ham and spam messages separately 
hamM = list(dfTrain[dfTrain.classifier == "ham"]["text"])
spamM = list(dfTrain[dfTrain.classifier == "spam"]["text"])
#train with "train"
t1=NaiveBayesForSpam()
t1.train(hamM,spamM)
#train with "train2"
t2=NaiveBayesForSpam()
t2.train2(hamM,spamM)

"""
Using the validation set, explore how each of the two classifi
ers performs out of sample.
"""
Vmessages=list(dfValidation["text"])
Vlabels=list(dfValidation["classifier"])

#explore with "train"
t1.score(Vmessages,Vlabels)

"""
the accuracy rate of train model is 0.9600.
"""

#explore with "train2"
t2.score(Vmessages,Vlabels)
"""
the accuracy rate of train2 model is 0.9680
"""

"""
Both classifiers have very high accuracy rate, however in this case train classifier performs 
slightly better than train2 classifier.
"""

"""
Why is the `train2' classifi
er faster? Why does it yield a better accuracy both on the
training and the validation set?
"""

"""
Because "train2" classifier first filter out the most popular spam word and then conduct the 
prediction. This makes the algorithm much faster since it does not need to examine each word
in spam category. 
"""

"""
How many false positives (ham messages classi
ed as spam messages) did you get
in your validation set? How would you change the code to reduce false positives at
the expense of possibly having more false negatives (spam messages classi
ed as ham
messages)?
"""
#number of False positives from train classifier
N1=25

#number of False positives from trian2 classifier 
N2=31

"""
I will change the "*20" to "*50" to decrease the number of words classified as spamkeyword
"""

"""
Run the `train2' classifi
er on the test set and report its performance using a confusion
matrix.
"""

Tmessages=list(dfValidation["text"])
Tlabels=list(dfValidation["classifier"])

t2.score(Tmessages,Tlabels)













