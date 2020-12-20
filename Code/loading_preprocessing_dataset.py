#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import os
import glob
import numpy as np
import cv2
from skimage import filters


# In[1]:


def prepare_dataset(Dataset_dir):
    
    SB_dir = os.path.join(Dataset_dir, 'Croped_SB/') #Satellite Bilder path
    GT_dir = os.path.join(Dataset_dir, 'Croped_GT/')# Ground Truths path


    
    SB_listnames=glob.glob(SB_dir+"*.png")#Satellite Bilder filenames
    GT_listnames=glob.glob(GT_dir+"*.png")# Ground Truths filenames
    
    GT_listnames.sort()
    SB_listnames.sort()
    
    print("Satellite Directory:",SB_dir)
    print('Number of  ground truths:',len(SB_listnames))
    print("") 
    print("Ground Truths Directory:",GT_dir)
    print('Number der satellien images:',len(GT_listnames))

    print("*********************************************") 

    
    return SB_dir, GT_dir, SB_listnames, GT_listnames


# In[2]:


def gaussian_filter(image, sigma = 2):
    img = np.copy(image)
    blur = filters.gaussian(img, sigma=sigma)
    return blur


# In[3]:


def binary(image, threshold, max_value = 1):
    img = np.copy(image)
    (t,masklayer) = cv2.threshold(img,threshold,max_value,cv2.THRESH_BINARY)
    return masklayer


# In[4]:


def find_threshold_otsu(image):
    t = filters.threshold_otsu(image)
    return t


# In[5]:


def load_image(s_image_path, gt_image_path):
    
    s_image_path=str(s_image_path).split("'")[1]
    gt_image_path=str(gt_image_path).split("'")[1]

    #print(s_image_path, gt_image_path)   
    s_image =cv2.imread(str(s_image_path))
    s_image= cv2.cvtColor(s_image,cv2.COLOR_BGR2RGB)

    #GT_path=str(GT_path).split("'")[1]
    gt_image =cv2.imread(str(gt_image_path))
    gt_image= cv2.cvtColor(gt_image,cv2.COLOR_BGR2RGB)



       
    if np.random.random() > 0.5:
     # random mirroring
        s_image = cv2.flip( s_image, -1);
        gt_image = cv2.flip( gt_image, -1);


        
    gt_image= cv2.cvtColor(gt_image,cv2.COLOR_RGB2GRAY)
     
        
    guass_img = gaussian_filter(gt_image, sigma =2)
    threshold =  find_threshold_otsu(guass_img)
    gt_image = binary(guass_img, threshold, max_value = 1)
    
    s_image = s_image / 255.0

    s_image = tf.cast(s_image, tf.float32)
    gt_image = tf.cast(gt_image, tf.float32)
    gt_image = tf.expand_dims( gt_image, 2)    


    return s_image,gt_image


# In[ ]:


def create_dataset(sb_list, gt_list, buffer_size =  2000, number_batche = 250 ):
    
    sb_dataset = tf.data.Dataset.from_tensor_slices(sb_list)
    gt_dataset = tf.data.Dataset.from_tensor_slices(gt_list)
    dataset = tf.data.Dataset.zip((sb_dataset, gt_dataset))
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(lambda x,y: tf.py_function(load_image, [x, y], [tf.float32,tf.float32]))
    dataset = dataset.batch(number_batche)
    
    return dataset

