
# coding: utf-8

# In[1]:

import sys
import pickle


# In[2]:


import numpy as np


# In[3]:


from sklearn.preprocessing import StandardScaler as SCR


# In[4]:


from sklearn.linear_model import LogisticRegression as LR


# In[5]:


from sklearn.svm import SVC


# In[6]:


from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout


# In[7]:


test_X = np.loadtxt(sys.argv[5], skiprows=1, delimiter=',')


# In[8]:


drop_Xt = np.c_[test_X[:, :10], test_X[:, 11:]]


# In[9]:


with open('scr.pickle', 'rb') as f:
    scr = pickle.load(f)


# In[10]:


prepro_Xt = scr.transform(drop_Xt)


# In[11]:


np.random.seed(5)


# In[12]:


with open('lr.pickle', 'rb') as f:
    lr = pickle.load(f)


# In[13]:


test_Y = lr.predict(prepro_Xt)


# In[14]:


with open('svc.pickle', 'rb') as f:
    svc = pickle.load(f)


# In[15]:


test_Y_SVC = svc.predict(prepro_Xt)


# In[16]:


model = load_model('nn')


# In[17]:


test_Y_NET = model.predict_classes(prepro_Xt).reshape(-1)


# In[18]:


test_Y_ENS = np.add(np.add(test_Y, test_Y_SVC), test_Y_NET) / 3
test_Y_ENS = np.round_(test_Y_ENS)


# In[19]:


with open(sys.argv[6], 'w+') as f:
    f.write('id,label\n')
    for i, v in enumerate(test_Y_ENS, 1):
        f.write('%d,%d\n' % (i, v))

