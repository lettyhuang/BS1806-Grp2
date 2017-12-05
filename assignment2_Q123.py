# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:24:02 2017

@author: letty
"""

import pandas as pd
import numpy as np
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
#


"""
Use your training set to train the classi
ers `train' and `train2'. Note that the interfaces
of our classi
ers require you to pass the ham and spam messages separately.
"""





