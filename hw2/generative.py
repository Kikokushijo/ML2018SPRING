
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


drop_X = (drop_X - mean) / (var ** 0.5)
drop_Y = drop_Y[:]
drop_Xt = (drop_Xt - mean) / (var ** 0.5)


# In[6]:


drop_X_0 = drop_X[drop_Y == 0]
drop_X_1 = drop_X[drop_Y == 1]
num_0, num_1 = len(drop_X_0), len(drop_X_1)
p_0 = num_0 / (num_0 + num_1)
p_1 = num_1 / (num_0 + num_1)
# print(p_0, p_1)


# In[7]:


var_0 = np.std(drop_X_0, axis=0)
var_1 = np.std(drop_X_1, axis=0)

mean_0 = np.mean(drop_X_0, axis=0).reshape(-1, 1)
mean_1 = np.mean(drop_X_1, axis=0).reshape(-1, 1)


# In[8]:


# print(mean_0.shape, mean_1.shape)


# In[9]:


cov_0 = np.cov(drop_X_0.T)
cov_1 = np.cov(drop_X_1.T)
cov = p_0 * cov_0 + p_1 * cov_1
# print(cov_0.shape, cov_1.shape, cov.shape)


# In[10]:


cov_inv = np.linalg.inv(cov)
cov_det = np.linalg.det(cov)


# In[11]:


# print(cov_inv, cov_det)


# In[12]:


def predict(vec):
    w = np.dot((mean_0 - mean_1).T, cov_inv)
    b = -0.5 * (np.dot(np.dot(mean_0.T, cov_inv), mean_0) - np.dot(np.dot(mean_1.T, cov_inv), mean_1)) + np.log(num_0 / num_1)
    z = (np.dot(w, vec) + b)[0][0]
    return int(z > 0)
#     return 1 / (1 + np.exp(-z))


# In[13]:


test_Y = [predict(vec) for vec in drop_Xt]


# In[14]:


with open(sys.argv[6], 'w+') as f:
    f.write('id,label\n')
    for i, v in enumerate(test_Y, 1):
        f.write('%d,%d\n' % (i, v))

