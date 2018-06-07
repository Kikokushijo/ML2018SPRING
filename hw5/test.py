
# coding: utf-8

# In[1]:


from gensim.models import word2vec
from gensim.parsing.porter import PorterStemmer


# In[2]:


import numpy as np


# In[3]:


import os
import re
import pickle
import string
import sys

testing_file = sys.argv[1]
predict_file = sys.argv[2]

# In[4]:


embed_size = 300


# In[5]:


with open('mapfile.pickle', 'rb') as f:
    mapfile = pickle.load(f)


# In[6]:


ps = PorterStemmer()


# In[7]:


table = str.maketrans({key: None for key in string.punctuation})


# In[8]:


def preprocess(s):
    for a, b in mapfile:
        s = s.replace(a, b)
    s = re.sub(r'\d+', '1', s)
    for same_char in re.findall(r'((\w)\2{2,})', s):
        s = s.replace(same_char[0], same_char[1])
    s = s.translate(table)
    s = ps.stem_sentence(s)
    return s


# In[9]:


test_sentences = []
with open(testing_file, 'r') as f:
    for line in f:
        idx, *sent = line.strip().split(',')
        sent = ','.join(sent)
        try:
            idx = int(idx)
            sent = preprocess(sent)
            test_sentences.append(sent.split())
        except:
            continue


# In[10]:


model = word2vec.Word2Vec.load('word2vec_all_0606.model')


# In[11]:


max_length = 37


# In[12]:


def sentences_generator(sentences, labels=None, batch_size=256):
    pos = 0
    while True:
        if labels is not None:
            label_batch = np.zeros((batch_size, 2))
        sent_batch = np.zeros((batch_size, max_length, embed_size))
        for id_sent in range(batch_size):
            if labels is not None:
                label = labels[pos]
                label_batch[id_sent][label] = 1.0
            sent = sentences[pos]
            id_word, src_word = max_length-1, len(sent)-1
            while id_word >= 0 and src_word >= 0:
                if sent[src_word] in model.wv:
                    sent_batch[id_sent][id_word][:] = model.wv[sent[src_word]]
                    id_word -= 1
                src_word -= 1
            pos = (pos + 1) % len(sentences)
        if labels is not None:
            yield sent_batch, label_batch
        else:
            yield sent_batch


# In[13]:


from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Masking, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional, Activation
from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


# In[14]:


class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})


# In[15]:


RNNmodel = load_model('rnn_strong_semi_0606')


# In[16]:


Yt = RNNmodel.predict_generator(sentences_generator(test_sentences, None, 200), len(test_sentences)//200, verbose=1)


# In[17]:


ans = np.argmax(Yt, axis=1)


# In[18]:


with open(predict_file, 'w+') as f:
    f.write('id,label\n')
    for idx, a in enumerate(ans):
        f.write('%d,%d\n' % (idx, a))

