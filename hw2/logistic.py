
# coding: utf-8

# In[1]:

import sys
import numpy as np


# In[2]:


train_X = np.loadtxt(sys.argv[3], skiprows=1, delimiter=',')
train_Y = np.loadtxt(sys.argv[4], delimiter=',')
test_X = np.loadtxt(sys.argv[5], skiprows=1, delimiter=',')


# In[3]:


drop_X = np.c_[train_X[:, :10], train_X[:, 11:]]
drop_Y = train_Y[:]
drop_Xt = np.c_[test_X[:, :10], test_X[:, 11:]]


# In[4]:


mean = np.loadtxt('scaler_mean')
var = np.loadtxt('scaler_var')


# In[5]:


prepro_X = (drop_X - mean) / (var ** 0.5)
prepro_Y = drop_Y[:]
prepro_Xt = (drop_Xt - mean) / (var ** 0.5)


# In[6]:


# print(prepro_X[0][:10])


# In[7]:


def shuffle_train_data(X, Y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)


# In[8]:


np.random.seed(5)


# In[9]:


shuffle_train_data(prepro_X, prepro_Y)


# In[10]:


iter_times = 1000000
eta = 1e-3


# In[11]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def acc(X, w, y):
    return (np.sign(np.dot(X, w) * y) + 1) / 2


# In[12]:


TRAIN_SIZE, TRAIN_FEAT = prepro_X.shape
TEST_SIZE, TEST_FEAT = prepro_Xt.shape
X = np.c_[np.ones((TRAIN_SIZE, 1)), prepro_X]
Y = prepro_Y * 2 - 1
Xt = np.c_[np.ones((TEST_SIZE, 1)), prepro_Xt]


# In[13]:


Ein = []
w = np.zeros((TRAIN_FEAT + 1))
for _ in range(iter_times):
    idx = _ % TRAIN_SIZE
    grad = sum((sigmoid(np.dot(X[idx, :], w) * Y[idx]) * (-Y[idx])).reshape(-1, 1) * X[idx, :])
    w += eta * grad


# In[14]:


test_Y = (np.sign(np.dot(Xt, -w)) + 1) / 2


# In[15]:


with open(sys.argv[6], 'w+') as f:
    f.write('id,label\n')
    for i, v in enumerate(test_Y, 1):
        f.write('%d,%d\n' % (i, v))

