
# coding: utf-8

# In[1]:


import sys
import os
import numpy as np 
from skimage import io


# In[2]:


imgs = np.empty((415, 600, 600, 3))
for i in range(415):
    filename = os.path.join(sys.argv[1], ('%d.jpg' % i))
    imgs[i, :, :, :] = io.imread(filename)
imgs = imgs.reshape(415, -1).T
avg = np.mean(imgs, axis=1).reshape(-1, 1)


# In[3]:


img_vecs = imgs - avg
u, s, vh = np.linalg.svd(img_vecs, full_matrices=False)


# In[24]:


target = io.imread(os.path.join(sys.argv[1], sys.argv[2])).reshape(-1) - avg.reshape(-1)
M = np.dot(target, u[:,:4])
M = np.sum(M * u[:,:4], axis=1) + avg.reshape(-1)
M -= np.min(M)
M /= np.max(M)
M = (M * 255.0).astype(np.uint8).reshape(600, 600, 3)
io.imsave('reconstruction.png', M)

