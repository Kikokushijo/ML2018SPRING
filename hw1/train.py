
# coding: utf-8

# In[1]:


feat2id = {'AMB_TEMP':0, 'CH4':1, 'CO':2, 'NMHC':3, 'NO':4, 'NO2':5, 'NOx':6, 'O3':7, 'PM10':8, 'PM2.5':9,
           'RAINFALL':10, 'RH':11, 'SO2':12, 'THC':13, 'WD_HR':14, 'WIND_DIREC':15 , 'WIND_SPEED':16, 'WS_HR':17}


# In[2]:


def clean_train(inputs='train.csv'):
    dataset = {}
    with open(inputs, encoding='BIG5') as f:
        for index, line in enumerate(f):
            if index:
                date, pos, feat, *val = line.strip().split(',')
#                 if feat == 'RAINFALL' or feat == 'WIND_DIREC':
#                     continue
                year, month, day = date.split('/')
                month, val = int(month)-1, [(0.0 if i == 'NR' else float(i)) for i in val]
                if month not in dataset:
                    dataset[month] = {}
                if feat not in dataset[month]:
                    dataset[month][feat] = []
                dataset[month][feat].extend(val)
    return dataset


# In[3]:


DS = clean_train()


# In[4]:


import numpy as np
import math


# In[5]:


d = np.empty((12, 480, 18))
for month, data in DS.items():
    for feat, val in data.items():
        if feat in feat2id:
            d[month, :, feat2id[feat]] = val


# In[6]:


X = np.empty((12 * 471, 9, 18))
Y = np.empty(12 * 471)


# In[7]:


index = 0
for m in range(12):
    for i in range(471):
        X[index, :, :] = d[m, i:i+9, :]
        Y[index] = d[m][i+9][feat2id['PM2.5']]
        index += 1
X = X[:index, :, :]
Y = Y[:index]
# print(X.shape)
# print(Y.shape)
# print(X[:5])
# print(Y[:5])


# In[8]:


X = X.reshape((12 * 471, -1))
X = np.c_[X, np.ones(5652)]
# print(X[:5])
# print(Y[:5])


# In[11]:


w = np.zeros(len(X[0]))
lr = 10
iter_times = 10000


# In[17]:


Xt = X.transpose()
s = np.zeros(len(X[0]))

for i in range(iter_times):
    diff = np.dot(X, w) - Y
    grad = np.dot(Xt, diff)
    s += grad ** 2
    ada = np.sqrt(s)
    w -= lr * grad / ada


# In[ ]:


# print(X.shape, Y.shape, w.shape)


# In[ ]:


np.save('model.npy', w)
# w = np.load('model.npy')