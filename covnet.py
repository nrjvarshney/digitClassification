
# coding: utf-8

# In[1]:

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# In[2]:

# a convnet takes as input tensors of shape (image_height, image_width,image_channels)
model.summary()


# ## output of every Conv2D and MaxPooling2D layer is a 3D tensor of shape (height, width, channels)
# 
# ## The width and height dimensions tend to shrink as you go deeper in the network
# 
# ## The number of channels is controlled by the firstargument passed to the Conv2D layers (32 or 64).
# 
# ## The next step is to feed the last output tensor (of shape (3, 3, 64) ) into a densely connected classifier network like those you’re already familiar with: a stack of Dense layers.
# 
# ## These classifiers process vectors, which are 1D , whereas the current output is a 3D tensor. First we have to flatten the 3D outputs to 1D , and then add a few Dense layers on top.

# In[3]:

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


# In[4]:

model.summary()


# # Training covnet on MNIST images

# In[ ]:

from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))

train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))

test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)

test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)


# In[6]:

test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[7]:

test_acc


# # Whereas the densely connected network from chapter 2 had a test accuracy of 97.8%, the basic convnet has a test accuracy of 99.3%: we decreased the error rate by 68% (relative).
