#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
from tensorflow.keras.layers import  Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import SpatialDropout2D, Permute, Activation, Reshape, PReLU
from tensorflow.keras.layers import concatenate, add, Input
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model

import os


# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# In[3]:



def initial_block(inp, number_filter = 13, filter_size = (3,3), stride = (2,2)):
    
    conv = Conv2D(number_filter, filter_size, padding = "same", strides = stride ) (inp) 
    max_pool = MaxPooling2D() (inp)
    merged = concatenate ([conv,max_pool], axis = 3)
    batch = BatchNormalization(momentum = 0.1)(merged)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    output = PReLU(shared_axes = [1, 2])(batch)
    return output


# In[4]:


def bottleneck(inp, number_filter = 32, internal_scale = 4, asymmetric = 0, dilated = 0, downsample = False, dropout_rate = 0.1):
    
    # main branch
    internal = number_filter // internal_scale
    encoder = inp
    
    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = Conv2D(internal,(input_stride, input_stride), (input_stride,input_stride) , use_bias =  False) (encoder)

    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum = 0.1) (encoder)
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    
    #con
    if not asymmetric and not dilated:
        encoder =  Conv2D(internal, (3,3),padding = "same" ) (encoder)
    elif asymmetric:
        encoder = Conv2D(internal, (1, asymmetric), use_bias = False, padding = "same") (encoder)
        encoder = Conv2D(internal, ( asymmetric, 1), padding = "same") (encoder)
    elif dilated:
        encoder = Conv2D(internal, (3,3), dilation_rate = (dilated, dilated), padding = "same") (encoder)
    
    # Batch normalization + PReLU
    encoder = BatchNormalization(momentum = 0.1) (encoder)
    encoder = PReLU(shared_axes=[1, 2])(encoder)
   
    # 1x1
    encoder = Conv2D(number_filter, (1, 1), use_bias=False)(encoder)
    encoder = BatchNormalization(momentum=0.1)(encoder)  # enet uses momentum of 0.1, keras default is 0.99
    encoder = SpatialDropout2D(dropout_rate)(encoder)
    
    other = inp
    if downsample:
        other = MaxPooling2D()(other)

        other = Permute((1, 3, 2))(other)

        pad_feature_maps = number_filter - inp.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = Permute((1, 3, 2))(other)
    
    encoder = add([encoder, other])
    encoder = PReLU(shared_axes=[1, 2])(encoder)
    
    return encoder


# In[5]:


def en_build(inp, dropout_rate=0.01):
    
    en_input = initial_block(inp)
    #print(en_input.shape)
    
    enc_layer1 = bottleneck(en_input, 32, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for _ in range(4):
        enc_layer1 = bottleneck(enc_layer1, 32, dropout_rate=dropout_rate)  # bottleneck 1.i
    #print(enc_layer1.shape)
    
    enc_layer2 = bottleneck(enc_layer1, 64, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for _ in range(2):
        enc_layer2 = bottleneck(enc_layer2, 64)  # bottleneck 2.1
        enc_layer2 = bottleneck(enc_layer2, 64, dilated=2)  # bottleneck 2.2
        enc_layer2 = bottleneck(enc_layer2, 64, asymmetric=5)  # bottleneck 2.3
        enc_layer2 = bottleneck(enc_layer2, 64, dilated=4)  # bottleneck 2.4
        enc_layer2 = bottleneck(enc_layer2, 64)  # bottleneck 2.5
        enc_layer2 = bottleneck(enc_layer2, 64, dilated=8)  # bottleneck 2.6
        enc_layer2 = bottleneck(enc_layer2, 64, asymmetric=5)  # bottleneck 2.7
        enc_layer2 = bottleneck(enc_layer2, 64, dilated=16)  # bottleneck 2.8
    #print(enc_layer2.shape)
    
    enc_layer3 = bottleneck(enc_layer2, 128, downsample=True)  # bottleneck 2.0
    # bottleneck 2.x and 3.x
    for _ in range(2):
        enc_layer3 = bottleneck(enc_layer3, 128)  # bottleneck 2.1
        enc_layer3 = bottleneck(enc_layer3, 128, dilated=2)  # bottleneck 2.2
        enc_layer3 = bottleneck(enc_layer3, 128, asymmetric=5)  # bottleneck 2.3
        enc_layer3 = bottleneck(enc_layer3, 128, dilated=4)  # bottleneck 2.4
        enc_layer3 = bottleneck(enc_layer3, 128)  # bottleneck 2.5
        enc_layer3 = bottleneck(enc_layer3, 128, dilated=8)  # bottleneck 2.6
        enc_layer3 = bottleneck(enc_layer3, 128, asymmetric=5)  # bottleneck 2.7
        enc_layer3 = bottleneck(enc_layer3, 128, dilated=16)  # bottleneck 2.8
    #print(enc_layer3.shape)
    
    
    return en_input, enc_layer1, enc_layer2, enc_layer3


# In[6]:


def de_bottleneck(encoder, decoder, number_filter, upsample=False, reverse_module=False):
    
    if decoder is not None:
        encoder = add([encoder, decoder])
        
    if  (number_filter // 4  == 0):

        internal = number_filter
    else:
        internal = number_filter // 4 
        
    

    x = Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    if not upsample:
        x = Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)

    x = Conv2D(number_filter, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != number_filter or upsample:
        other = Conv2D(number_filter, (1, 1), padding='same', use_bias=False)(other)
        other = BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = UpSampling2D(size=(2, 2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)

    return decoder


# In[7]:



def de_build(enc_layer3, enc_layer2, enc_layer1, en_input, number_class = 3):
    
    dec_1 = de_bottleneck(enc_layer3,None, 64, upsample=True, reverse_module=True)
    dec_1 = de_bottleneck(dec_1,None, 64)  # bottleneck 1.1
    dec_1 = de_bottleneck(dec_1,None, 64)  # bottleneck 1.2
    
    dec_2 = de_bottleneck(dec_1, enc_layer2, 32, upsample=True, reverse_module=True)  # bottleneck 5.0
    dec_2 = de_bottleneck(dec_2,None, 32)  # bottleneck 2.1
    
    dec_3 = de_bottleneck(dec_2, enc_layer1, 16, upsample=True, reverse_module=True)  # bottleneck 5.0
    dec_3 = de_bottleneck(dec_3,None, 16)  # bottleneck 3.1

    dec_4 = de_bottleneck(dec_3, en_input, number_class, upsample=True, reverse_module=True)  # bottleneck 5.0
    dec_4 = de_bottleneck(dec_4, None, number_class )  # bottleneck 4.1   
    dec_4 = de_bottleneck(dec_4, None, number_class )  # bottleneck 4.1    

    return dec_4


# In[8]:


def build_EUNet(number_classes, input_height=256, input_width=256):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    
    img_input = Input(shape=(input_height, input_width, 3))
    en_input, enc_layer1, enc_layer2, enc_layer3 = en_build(img_input, dropout_rate=0.01)    
    output = de_build(enc_layer3, enc_layer2, enc_layer1, en_input, number_class = number_classes)
    
    model = Model(img_input, output)
    

    return model


# In[9]:

