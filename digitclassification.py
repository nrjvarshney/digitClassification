
# coding: utf-8

# # The problem we’re trying to solve here is to classify grayscale images of handwritten digits (28 × 28 pixels) into their 10 categories (0 through 9).

# In[1]:


from keras.datasets import mnist


# In[16]:


(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()


# In[3]:


# The images are encoded as Numpy arrays, and the labels are an array of digits, ranging from 0 to 9.
train_images.shape


# In[6]:


len(train_images), len(train_labels)


# In[7]:


train_labels


# In[10]:


test_images.shape


# In[17]:


digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap = plt.cm.binary)
plt.show()


# In[8]:


from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation = 'relu', input_shape = (28*28, )))
network.add(layers.Dense(10, activation = 'softmax'))


# Here, our network consists of a sequence of two Dense layers, which are densely
# connected (also called fully connected) neural layers. The second (and last) layer is a
# 10-way softmax layer, which means it will return an array of 10 probability scores (summing
# to 1). Each score will be the probability that the current digit image belongs to
# one of our 10 digit classes

# A loss function—How the network will be able to measure its performance on
# the training data, and thus how it will be able to steer itself in the right direction.
# 
# 
# An optimizer—The mechanism through which the network will update itself
# based on the data it sees and its loss function.
# 
# 
# Metrics to monitor during training and testing—Here, we’ll only care about accuracy
# (the fraction of the images that were correctly classified).

# In[9]:


network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# # Transforming data

# In[10]:


train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# In[12]:


from keras.utils import to_categorical 

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[13]:


network.fit(train_images, train_labels, epochs =5, batch_size = 128)


# In[14]:


test_loss, test_acc = network.evaluate(test_images, test_labels)
test_acc

