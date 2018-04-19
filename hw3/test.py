
# coding: utf-8

# In[1]:


from keras.models import load_model


# In[2]:


import numpy as np


# In[3]:


from io import StringIO
import sys


# In[4]:


def preprocess(X, Y=None):
    X_ = X.reshape(-1, img_sizeX, img_sizeY, 1)
    if Y is not None:
        Y_ = np.zeros((len(X), cls_num))
        Y_[np.arange(len(X)), Y] = 1.0
        return X_, Y_
    return X_


# In[5]:


img_sizeX = 48
img_sizeY = 48
cls_num = 7


# In[6]:


test_name = sys.argv[1]
with open(test_name, 'r') as f:
    s = f.read().replace(',',' ')
    test_data = np.loadtxt(StringIO(s), skiprows=1, dtype=int)


# In[7]:


Xtest = test_data[:, 1:]


# In[8]:


pXtest = preprocess(Xtest)


# In[9]:


def output(model_name, predict_name=None):
    model = load_model(model_name)
    ans = model.predict(pXtest)
    ans = np.argmax(ans, axis=1)
    if predict_name is None:
        predict_name = 'ans' + model_name + '_test.csv'
    with open(predict_name, 'w+') as f:
        f.write('id,label\n')
        for idx, y in enumerate(ans):
            f.write('%d,%d\n' % (idx, y))


# In[10]:

if sys.argv[3] == 'public':
    output('04161753.h5', sys.argv[2])
elif sys.argv[3] == 'private':
    output('04170222.h5', sys.argv[2])

