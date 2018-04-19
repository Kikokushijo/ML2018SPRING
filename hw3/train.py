
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[2]:


from keras.models import load_model


# In[3]:


from keras.layers import Concatenate, Add


# In[4]:


from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[5]:


from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np


# In[6]:


from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model
# import six
import sys


# In[7]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Masking, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


# In[8]:


img_sizeX = 48
img_sizeY = 48
cls_num = 7


# In[10]:


from io import StringIO
# print(sys.argv[0])
with open(sys.argv[1], 'r') as f:
    s = f.read().replace(',',' ')
    train_data = np.loadtxt(StringIO(s), skiprows=1, dtype=int)


# In[11]:


split_rate = 0.9
split_num = int(len(train_data) * split_rate)


# In[12]:


np.random.seed(666)
np.random.shuffle(train_data)
X = train_data[:, 1:]
Y = train_data[:, 0]
Xt = X[:split_num, :]
Yt = Y[:split_num]
Xv = X[split_num:, :]
Yv = Y[split_num:]
# del train_data, X, Y


# In[13]:


def preprocess(X, Y=None):
    X_ = X.reshape(-1, img_sizeX, img_sizeY, 1)
    if Y is not None:
        Y_ = np.zeros((len(X), cls_num))
        Y_[np.arange(len(X)), Y] = 1.0
        return X_, Y_
    return X_


# In[14]:


pXt, pYt = preprocess(Xt, Yt)
pXv, pYv = preprocess(Xv, Yv)


# In[15]:


del train_data


# In[16]:


datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        data_format='channels_last')


# In[17]:


def output(model_name):
    model = load_model(model_name)
    ans = model.predict(pXtest)
    ans = np.argmax(ans, axis=1)
    with open('ans' + model_name + '.csv', 'w+') as f:
        f.write('id,label\n')
        for idx, y in enumerate(ans):
    #         print(idx, y, max(enumerate(y), key=lambda x:x[1])[0])
            f.write('%d,%d\n' % (idx, y))


# In[18]:


x = Input(shape=(48, 48, 1))

conv0_in = Conv2D(filters=64, kernel_size=(5, 5), strides=2, padding='same')(x)
conv0_bn = BatchNormalization()(conv0_in)
conv0_out = Activation('relu')(conv0_bn)

conv1_in = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same' )(conv0_out)
conv1_bn = BatchNormalization()(conv1_in)
conv1_out = Activation('relu')(conv1_bn)
conv2_in = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(conv1_out)
conv2_bn = BatchNormalization()(conv2_in)
conv2_res = Activation('relu')(conv2_bn)
conv2_out = Add()([conv0_out, conv2_res])

conv3_in = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(conv2_out)
conv3_bn = BatchNormalization()(conv3_in)
conv3_out = Activation('relu')(conv3_bn)
conv4_in = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(conv3_out)
conv4_bn = BatchNormalization()(conv4_in)
conv4_res = Activation('relu')(conv4_bn)
conv4_out = Add()([conv2_out, conv4_res])

conv5_in = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(conv4_out)
conv5_bn = BatchNormalization()(conv5_in)
conv5_out = Activation('relu')(conv5_bn)
conv6_in = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(conv5_out)
conv6_bn = BatchNormalization()(conv6_in)
conv6_res = Activation('relu')(conv6_bn)
conv6_out = Add()([conv4_out, conv6_res])

conv7_in = Conv2D(filters=128, kernel_size=(3, 3), strides=2, padding='same')(conv6_out)
conv7_bn = BatchNormalization()(conv7_in)
conv7_out = Activation('relu')(conv7_bn)
conv8_in = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(conv7_out)
conv8_bn = BatchNormalization()(conv8_in)
conv8_out = Activation('relu')(conv8_bn)

conv9_in = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(conv8_out)
conv9_bn = BatchNormalization()(conv9_in)
conv9_out = Activation('relu')(conv9_bn)
conv10_in = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(conv9_out)
conv10_bn = BatchNormalization()(conv10_in)
conv10_res = Activation('relu')(conv10_bn)
conv10_out = Add()([conv8_out, conv10_res])

conv11_in = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(conv10_out)
conv11_bn = BatchNormalization()(conv11_in)
conv11_out = Activation('relu')(conv11_bn)
conv12_in = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(conv11_out)
conv12_bn = BatchNormalization()(conv12_in)
conv12_res = Activation('relu')(conv12_bn)
conv12_out = Add()([conv10_out, conv12_res])

conv13_in = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(conv12_out)
conv13_bn = BatchNormalization()(conv13_in)
conv13_out = Activation('relu')(conv13_bn)
conv14_in = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')(conv13_out)
conv14_bn = BatchNormalization()(conv14_in)
conv14_res = Activation('relu')(conv14_bn)
conv14_out = Add()([conv12_out, conv14_res])

conv15_in = Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same')(conv14_out)
conv15_bn = BatchNormalization()(conv15_in)
conv15_out = Activation('relu')(conv15_bn)
conv16_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv15_out)
conv16_bn = BatchNormalization()(conv16_in)
conv16_out = Activation('relu')(conv16_bn)

conv17_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv16_out)
conv17_bn = BatchNormalization()(conv17_in)
conv17_out = Activation('relu')(conv17_bn)
conv18_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv17_out)
conv18_bn = BatchNormalization()(conv18_in)
conv18_res = Activation('relu')(conv18_bn)
conv18_out = Add()([conv16_out, conv18_res])

conv19_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv18_out)
conv19_bn = BatchNormalization()(conv19_in)
conv19_out = Activation('relu')(conv19_bn)
conv20_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv19_out)
conv20_bn = BatchNormalization()(conv20_in)
conv20_res = Activation('relu')(conv20_bn)
conv20_out = Add()([conv18_out, conv20_res])

conv21_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv20_out)
conv21_bn = BatchNormalization()(conv21_in)
conv21_out = Activation('relu')(conv21_bn)
conv22_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv21_out)
conv22_bn = BatchNormalization()(conv22_in)
conv22_res = Activation('relu')(conv22_bn)
conv22_out = Add()([conv20_out, conv22_res])

conv23_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv22_out)
conv23_bn = BatchNormalization()(conv23_in)
conv23_out = Activation('relu')(conv23_bn)
conv24_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv23_out)
conv24_bn = BatchNormalization()(conv24_in)
conv24_res = Activation('relu')(conv24_bn)
conv24_out = Add()([conv22_out, conv24_res])

conv25_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv24_out)
conv25_bn = BatchNormalization()(conv25_in)
conv25_out = Activation('relu')(conv25_bn)
conv26_in = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')(conv25_out)
conv26_bn = BatchNormalization()(conv26_in)
conv26_res = Activation('relu')(conv26_bn)
conv26_out = Add()([conv24_out, conv26_res])

conv27_in = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same')(conv26_out)
conv27_bn = BatchNormalization()(conv27_in)
conv27_out = Activation('relu')(conv27_bn)
conv28_in = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(conv27_out)
conv28_bn = BatchNormalization()(conv28_in)
conv28_out = Activation('relu')(conv28_bn)

conv29_in = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(conv28_out)
conv29_bn = BatchNormalization()(conv29_in)
conv29_out = Activation('relu')(conv29_bn)
conv30_in = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(conv29_out)
conv30_bn = BatchNormalization()(conv30_in)
conv30_res = Activation('relu')(conv30_bn)
conv30_out = Add()([conv28_out, conv30_res])

conv31_in = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(conv30_out)
conv31_bn = BatchNormalization()(conv31_in)
conv31_out = Activation('relu')(conv31_bn)
conv32_in = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(conv31_out)
conv32_bn = BatchNormalization()(conv32_in)
conv32_res = Activation('relu')(conv32_bn)
conv32_out = Add()([conv30_out, conv32_res])

avg = AveragePooling2D(pool_size=(1, 1))(conv32_out)
flat = Flatten()(avg)
dense1 = Dense(256, activation='relu')(flat)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
dense4 = Dense(7, activation='softmax')(dense3)

model = Model(inputs=x, outputs=dense4)

print(model.summary())


# In[19]:


# pX, pY = preprocess(X, Y)
# gen = data_gen(pXt, pYt, gen=datagen, batch_size=28709)
cp = ModelCheckpoint('tmp.h5', monitor='val_acc', verbose=0, save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_generator = datagen.flow(pXt, pYt, batch_size=128)
# for i in range(500):
#     x, y = next(train_generator)
#     print(x[0], y[0], pXv[0], pYv[0])
#     print(X[:5])
#     break
#     x = x / 255.0 - 0.5
#     print(X[:5], Y[:5])
#     break
#     print(x.shape, y.shape)
#     print('Epoch %d:' % i)
#     model.fit(x, y, batch_size=128, epochs=1, verbose=1, validation_data=(pXv / 255.0 - 0.5, pYv), shuffle=True)
#     model.fit(x, y, batch_size=128, epochs=1, verbose=1, validation_data=(pXv, pYv), shuffle=True)
#     history = model.fit_generator(train_generator, callbacks=[cp], steps_per_epoch=(split_num // 128 + 1), \
#                                   samples_per_epoch=128, epochs=250, verbose=1, validation_data=(pXv, pYv))
history = model.fit_generator(train_generator, steps_per_epoch=(split_num // 128 + 1), epochs=2500,                                   verbose=1, callbacks=[cp], validation_data=(pXv, pYv))
#     if i % 5 == 0:
#     model.save('04161340_%d' % i)
# model.fit_generator(generator=data_gen(pXt, pYt, gen=datagen, batch_size=128),
#                     steps_per_epoch=160,
#                     epochs=1000, verbose=1,
#                     validation_data=data_gen2(pXv, pYv, gen=test_datagen, batch_size=8229),
#                     validation_steps=1)

