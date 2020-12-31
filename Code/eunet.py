#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
from tensorflow.keras.layers import  Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import SpatialDropout2D, Permute, Activation, Reshape, PReLU
from tensorflow.keras.layers import concatenate, add, Input
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
import datetime

# In[2]:



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
    
    enc_layer1 = initial_block(inp)
    #print(enc_layer1.shape)
    
    enc_layer2 = bottleneck(enc_layer1, 32, downsample=True, dropout_rate=dropout_rate)  # bottleneck 1.0
    for _ in range(3):
        enc_layer2 = bottleneck(enc_layer2, 32, dropout_rate=dropout_rate)  # bottleneck 1.i
    #print(enc_layer2.shape)
    
    enc_layer3 = bottleneck(enc_layer2, 64, downsample=True)  # bottleneck 2.0
    for _ in range(2):
        enc_layer3 = bottleneck(enc_layer3, 64)  # bottleneck 2.1
        enc_layer3 = bottleneck(enc_layer3, 64, dilated=2)  # bottleneck 2.2
        enc_layer3 = bottleneck(enc_layer3, 64, asymmetric=5)  # bottleneck 2.3
        enc_layer3 = bottleneck(enc_layer3, 64, dilated=4)  # bottleneck 2.4
        enc_layer3 = bottleneck(enc_layer3, 64)  # bottleneck 2.5
        #enc_layer3 = bottleneck(enc_layer3, 64, dilated=8)  # bottleneck 2.6
        #enc_layer3 = bottleneck(enc_layer3, 64, asymmetric=5)  # bottleneck 2.7
        #enc_layer3 = bottleneck(enc_layer3, 64, dilated=16)  # bottleneck 2.8
    #print(enc_layer3.shape)
    
    enc_layer4 = bottleneck(enc_layer3, 128, downsample=True)  # bottleneck 2.0
    for _ in range(1):
        enc_layer4 = bottleneck(enc_layer4, 128)  # bottleneck 3.1
        enc_layer4 = bottleneck(enc_layer4, 128, dilated=2)  # bottleneck 3.2
        enc_layer4 = bottleneck(enc_layer4, 128, asymmetric=5)  # bottleneck 3.3
        enc_layer4 = bottleneck(enc_layer4, 128, dilated=4)  # bottleneck 3.4
        enc_layer4 = bottleneck(enc_layer4, 128)  # bottleneck 3.5
        #enc_layer4 = bottleneck(enc_layer4, 128, dilated=8)  # bottleneck 3.6
        #enc_layer4 = bottleneck(enc_layer4, 128, asymmetric=5)  # bottleneck 3.7
        #enc_layer4 = bottleneck(enc_layer4, 128, dilated=16)  # bottleneck 3.8
    #print(enc_layer4.shape)
    

    
    return enc_layer1, enc_layer2, enc_layer3, enc_layer4#, enc_layer4


# In[6]:


def de_bottleneck(encoder, decoder, number_filter, upsample=False, reverse_module=False, last=False):
    
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
    elif last:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('sigmoid')(decoder)
    else:
        x = BatchNormalization(momentum=0.1)(x)
        decoder = add([x, other])
        decoder = Activation('relu')(decoder)
    
    return decoder


# In[7]:



def de_build(enc_layer4, enc_layer3, enc_layer2, enc_layer1, number_class = 3):
    
    dec_1 = de_bottleneck(enc_layer4,None, 64, upsample=True, reverse_module=True)
    dec_1_1 = de_bottleneck(dec_1,None, 64)  # bottleneck 1.1
    dec_1_2 = de_bottleneck(dec_1_1,None, 64)  # bottleneck 1.2


    dec_2 = de_bottleneck(dec_1_2, enc_layer3, 32, upsample=True, reverse_module=True)
    dec_2_1 = de_bottleneck(dec_2,None, 32)  # bottleneck 2.1
    dec_2_2 = de_bottleneck(dec_2_1,None, 32)  # bottleneck 2.2
    dec_2_3 = de_bottleneck(dec_2_2,None, 32)  # bottleneck 2.3
    dec_2_4 = de_bottleneck(dec_2_3,None, 32)  # bottleneck 2.4

    dec_3 = de_bottleneck(dec_2_4, enc_layer2, 16, upsample=True, reverse_module=True )
    dec_3_1 = de_bottleneck(dec_3,None, 16)  # bottleneck 3.1
    dec_3_2 = de_bottleneck(dec_3_1,None, 16)  # bottleneck 3.2
    dec_3_3 = de_bottleneck(dec_3_2,None, 16)  # bottleneck 3.3
    dec_3_4 = de_bottleneck(dec_3_3,None, 16)  # bottleneck 3.4
    dec_3_5 = de_bottleneck(dec_3_4,None, 16)  # bottleneck 3.5
    dec_3_6 = de_bottleneck(dec_3_5,None, 16)  # bottleneck 3.6


    dec_4 = de_bottleneck(dec_3_6, enc_layer1, number_class, upsample=True, reverse_module=True)
    dec_4_1 = de_bottleneck(dec_4, None, number_class)# bottleneck 5.1 
    dec_4_2 = de_bottleneck(dec_4_1, None, number_class)# bottleneck 5.2 
    dec_4_3 = de_bottleneck(dec_4_2, None, number_class)# bottleneck 5.3     
    dec_4_4 = de_bottleneck(dec_4_3, None, number_class)# bottleneck 5.4 
    dec_4_5 = de_bottleneck(dec_4_4, None, number_class)# bottleneck 5.5 
    dec_4_6 = de_bottleneck(dec_4_5, None, number_class)# bottleneck 5.6 
    dec_4_7 = de_bottleneck(dec_4_6, None, number_class)# bottleneck 5.7 
    dec_4_8 = de_bottleneck(dec_4_7, None, number_class, last=True)# bottleneck 5.8 
 
    
    return dec_4_8


# In[8]:


def build_EUNet(number_classes, input_height=256, input_width=256):
    assert input_height % 32 == 0
    assert input_width % 32 == 0
    
    img_input = Input(shape=(input_height, input_width, 3))
    enc_layer1, enc_layer2, enc_layer3, enc_layer4 = en_build(img_input, dropout_rate=0.01)    
    output = de_build( enc_layer4, enc_layer3, enc_layer2, enc_layer1, number_class = number_classes)
    
    model = Model(img_input, output)

    return model


# In[9]:
def get_callbacks():
    
    timestr = datetime.datetime.now().strftime("(%m-%d-%Y , %H:%M:%S)")
    model_dir = os.path.join('./models','E_UNet_{}'.format(timestr))
    checkpoint = ModelCheckpoint(model_dir, monitor='val_loss', verbose=2, 
                                 save_best_only=True, mode='min', save_weights_only = False)

    log_dir = os.path.join('./logs','E_UNet_{}'.format(timestr))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

    callbacks_list = [checkpoint, tensorboard] 
    
    return callbacks_list
