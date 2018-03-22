import numpy as np
import sys

feat2id = {'AMB_TEMP':0, 'CH4':1, 'CO':2, 'NMHC':3, 'NO':4, 'NO2':5, 'NOx':6, 'O3':7, 'PM10':8, 'PM2.5':9,
           'RAINFALL':10, 'RH':11, 'SO2':12, 'THC':13, 'WD_HR':14, 'WIND_DIREC':15 , 'WIND_SPEED':16, 'WS_HR':17}


def clean_test(inputs=sys.argv[1]):
    dataset = np.empty((260, 9, 18))
    with open(inputs, encoding='BIG5') as f:
        for index, line in enumerate(f):
            index, feat, *val = line.strip().split(',')
            _, real_id = index.split('_')
            real_id = int(real_id)
            val = [(0.0 if i == 'NR' else float(i)) for i in val]
            if feat in feat2id:
                dataset[real_id, :, feat2id[feat]] = val
    return dataset


# In[19]:
test_dataset = clean_test()
w = np.load('model.npy')

# In[20]:


X_train = test_dataset.reshape(260, -1)
X_train = np.c_[X_train, np.ones(260)]


# In[21]:


with open(sys.argv[2], 'w+') as f:
    f.write('id,value\n')
    for i, v in enumerate(X_train):
        f.write('id_%d,%f\n' % (i, np.dot(w, v)))
