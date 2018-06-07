
# coding: utf-8

# In[ ]:


from gensim.models import word2vec
from gensim.parsing.porter import PorterStemmer


# In[ ]:


import numpy as np


# In[ ]:


import os
import re
import pickle
import string
import sys

training_file = sys.argv[1]
nonlabel_file = sys.argv[2]

# In[ ]:


embed_size = 300


# In[ ]:


with open('mapfile.pickle', 'rb') as f:
    mapfile = pickle.load(f)
print('Read Mapfile...OK')

# In[ ]:


ps = PorterStemmer()


# In[ ]:


table = str.maketrans({key: None for key in string.punctuation})


# In[ ]:


def preprocess(s):
    for a, b in mapfile:
        s = s.replace(a, b)
    s = re.sub(r'\d+', '1', s)
    for same_char in re.findall(r'((\w)\2{2,})', s):
        s = s.replace(same_char[0], same_char[1])
    s = s.translate(table)
    s = ps.stem_sentence(s)
    return s


# In[ ]:


train_labels, train_sentences = [], []
with open(training_file, 'r') as f:
    for line in f:
        label, sent = line.strip().split(' +++$+++ ')
        sent = preprocess(sent)
        train_labels.append(int(label))
        train_sentences.append(sent.split())
print('Read Train Sentences...OK')

# In[ ]:


nl_sentences = []
with open(nonlabel_file, 'r') as f:
    for line in f:
        sent = line.strip()
        sent = preprocess(sent)
        nl_sentences.append(sent.split())
print('Read NonLabel Sentences...OK')

# In[ ]:


if os.path.isfile('word2vec.model'):
    model = word2vec.Word2Vec.load('word2vec.model')
else:
    model = word2vec.Word2Vec(train_sentences+nl_sentences, size=embed_size, workers=8, min_count=1, window=5)
    model.save('word2vec.model')
print('Load Word2Vec Model...OK')

# In[ ]:


max_length = max([len(s) for s in train_sentences])


# In[ ]:


from sklearn.utils import shuffle
train_labels, train_sentences = shuffle(train_labels, train_sentences)
split_rate = 0.93
split_num = int(len(train_labels) * split_rate)
valid_labels, valid_sentences = train_labels[split_num:], train_sentences[split_num:]
train_labels, train_sentences = train_labels[:split_num], train_sentences[:split_num]

# In[ ]:


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


# In[ ]:


tmpgen = sentences_generator(valid_sentences, valid_labels, len(valid_sentences))
Xv, Yv = next(tmpgen)
Xv.shape, Yv.shape


# In[ ]:


from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Masking, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Bidirectional, Activation
from keras.layers.recurrent import GRU, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K


# In[ ]:


class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})


# In[ ]:


RNNmodel = Sequential()
RNNmodel.add(Bidirectional(LSTM(128, activation="tanh", dropout=0.2, return_sequences=True, kernel_initializer='he_uniform'), input_shape=(max_length, embed_size)))
RNNmodel.add(Bidirectional(LSTM( 64, activation="tanh", dropout=0.2, kernel_initializer='he_uniform')))
RNNmodel.add(BatchNormalization())
RNNmodel.add(Dense(256, activation=swish))
RNNmodel.add(Dropout(0.3))
RNNmodel.add(Dense(128, activation=swish))
RNNmodel.add(Dropout(0.3))
RNNmodel.add(Dense( 64, activation=swish))
RNNmodel.add(Dropout(0.3))
RNNmodel.add(Dense(  2, activation='softmax'))
RNNmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(RNNmodel.summary())


# In[ ]:


cp = ModelCheckpoint('rnn', monitor='val_acc', verbose=0, save_best_only=True)


# In[ ]:


batch_size = 1000
gen = sentences_generator(sentences=train_sentences, labels=train_labels, batch_size=batch_size)
history = RNNmodel.fit_generator(gen, steps_per_epoch=(split_num // batch_size), epochs=20, callbacks=[cp], verbose=1, validation_data=(Xv, Yv))


# In[ ]:


if os.path.isfile('rnn'):
    get_custom_objects().update({'swish': Activation(swish)})
    RNNmodel = load_model('rnn')
