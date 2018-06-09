
# coding: utf-8

# In[1]:

import sys
test_file = sys.argv[1]
pred_file = sys.argv[2]
mv_file = sys.argv[3]
us_file = sys.argv[4]

import pandas as pd
# movies = pd.read_csv(mv_file, delimiter='::', engine='python')


# In[2]:


test = pd.read_csv(test_file)


# In[3]:


import pickle
with open('u2id', 'rb') as f, open('m2id', 'rb') as g:
    u2id = pickle.load(f)
    m2id = pickle.load(g)


# In[4]:


from keras.layers import Input, Embedding, Flatten, Dropout, Lambda, Dot, Add
from keras.regularizers import l2
from keras.initializers import Zeros
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K
import numpy as np


# In[5]:


mean, std = 3.5817120860388076, 1.116897661146206
get_custom_objects().update({'mean': mean})
get_custom_objects().update({'std': std})


# In[6]:


X_u_test = np.array([u2id[x] for x in test['UserID']])
X_m_test = np.array([m2id[x] for x in test['MovieID']])
model = load_model('model_simple0.h5')
Y_test = model.predict([X_u_test, X_m_test]).reshape(-1)


# In[7]:


Y_test = np.clip(Y_test, 1, 5)


# In[8]:


with open(pred_file, 'w+') as f:
    f.write('TestDataID,Rating\n')
    for i, r in enumerate(Y_test, 1):
        f.write('%d,%f\n' % (i, r))

