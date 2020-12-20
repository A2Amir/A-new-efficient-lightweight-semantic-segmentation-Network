#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras import backend as K


# In[2]:


def Weighted_Categorical_CrossEntropy(WEIGHTS):
        #WEIGHTS = { 'weights': [1.0,2.0]}

    weights = WEIGHTS['weights']
    weights = tf.Variable(weights,dtype=tf.float32)
    def loss_(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -tf.reduce_sum(loss, -1)
        return loss
    return loss_
 


# In[ ]:


def binary_crossentropy(y_true, y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
    return K.sum(class_loglosses )

