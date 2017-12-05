# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:24:02 2017

@author: letty
"""

import pandas as pd
import numpy as np
import nltk as nl
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



