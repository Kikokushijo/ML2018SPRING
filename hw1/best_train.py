import sys

# print(sys.argv[1], sys.argv[2])

newfeat2id = {'PM2.5':0, 'PM10':1}


# In[2]:


def clean_train(inputs='train.csv'):
    dataset = {}
    with open(inputs, encoding='BIG5') as f:
        for index, line in enumerate(f):
            if index:
                date, pos, feat, *val = line.strip().split(',')
                if feat == 'RAINFALL' or feat == 'WIND_DIREC':
                    continue
                year, month, day = date.split('/')
                month, val = int(month)-1, [float(i) for i in val]
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


# In[5]:


d = np.empty((12, 480, 2))
for month, data in DS.items():
    for feat, val in data.items():
        if feat in newfeat2id:
            d[month, :, newfeat2id[feat]] = val


# In[6]:


X = np.empty((12 * 471, 9, 2))
Y = np.empty(12 * 471)


# In[7]:


index = 0
for m in range(12):
    for i in range(471):
        if np.any(d[m, i:i+10, 0] > 120) or np.any(d[m, i:i+10, 0] < 0):
            continue
        X[index, :, :] = d[m, i:i+9, :]
        Y[index] = d[m][i+9][0]
        index += 1
X = X[:index, :, :]
Y = Y[:index]


# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


Xn = X.reshape((-1, 18))


# In[22]:


lr = LinearRegression(normalize=True)


# In[23]:


lr.fit(Xn, Y)


import pickle
with open('best.pickle', 'wb') as f:
    pickle.dump(lr, f)