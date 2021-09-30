
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Model
from keras import initializers
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Dropout, Add, Activation, SpatialDropout3D, BatchNormalization, Concatenate
from keras import backend as K
import math


# In[4]:


def frst_blck(inp, nfilter, k1, k2, k3, actvfc): 
    #print('input shape', inp.shape)
    x1=Conv3D(filters=nfilter, kernel_size=(k1,k1,k1), padding='same', kernel_initializer='he_uniform')(inp)
    x1=BatchNormalization()(x1)
    x1= Activation(actvfc)(x1)
    
    x2=Conv3D(filters=nfilter, kernel_size=(k2,k2,k2), padding='same', kernel_initializer='he_uniform')(x1) 
    x2=BatchNormalization()(x2)
    x2= Activation(actvfc)(x2)
    
    x3=Conv3D(filters=nfilter, kernel_size=(k3,k3,k3), padding='same', kernel_initializer='he_uniform')(x2) 
    x3=BatchNormalization()(x3)
    x3= Activation(actvfc)(x3)
    #print('input shape before add', x3.shape)
    
    res= Add()([x1, x3])
    #print('input after add', res.shape)
    return res

def last_blck(frst_block, previous_block, nfilter, k1, k2, k3, actvfc, p, add): 
    #print('frst_block.shape ', frst_block.shape)
    #print('previous_block.shape ', previous_block.shape)
    
    previous_block=UpSampling3D(size=(2,2,2))(previous_block)
    previous_block=Conv3D(filters=nfilter, kernel_size=(2,2,2), padding='same', activation=actvfc, 
                          kernel_initializer='he_uniform')(previous_block)
    if add:
        x=Add()([previous_block, frst_block])
    else: 
        x=Concatenate()([previous_block, frst_block])
        
    x= SpatialDropout3D(p)(x)
    
    x1=Conv3D(filters=nfilter, kernel_size=(k1,k1,k1), padding='same', kernel_initializer='he_uniform')(x)
    x1=BatchNormalization()(x1)
    x1= Activation(actvfc)(x1)
    
    x2=Conv3D(filters=nfilter, kernel_size=(k2,k2,k2), padding='same', kernel_initializer='he_uniform')(x1) 
    x2=BatchNormalization()(x2)
    x2= Activation(actvfc)(x2)
    
    x3=Conv3D(filters=nfilter, kernel_size=(k3,k3,k3), padding='same', kernel_initializer='he_uniform')(x2) 
    x3=BatchNormalization()(x3)
    x3= Activation(actvfc)(x3)
    
    #print('last shape before add', x3.shape)
    
    res= Add()([x1, x3])
    output= Conv3D(filters=1, kernel_size=1, activation='sigmoid', kernel_initializer='he_uniform' )(res) 
    #print('last after add', output.shape)
    return output


# In[5]:


def res_downsampling(previous_block, nfilter, k1, k2, k3, actvfc, p): 
    
    #print('previous_block.shape ', previous_block.shape)
    x= MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2))(previous_block) 
    x= SpatialDropout3D(p)(x)
    
    x1=Conv3D(filters=nfilter, kernel_size=(k1,k1,k1), padding='same', kernel_initializer='he_uniform')(x)
    x1=BatchNormalization()(x1)
    x1= Activation(actvfc)(x1)
    
    x2=Conv3D(filters=nfilter, kernel_size=(k2,k2,k2), padding='same', kernel_initializer='he_uniform')(x1) 
    x2=BatchNormalization()(x2)
    x2= Activation(actvfc)(x2)
    
    x3=Conv3D(filters=nfilter, kernel_size=(k3,k3,k3), padding='same', kernel_initializer='he_uniform')(x2) 
    x3=BatchNormalization()(x3)
    x3= Activation(actvfc)(x3)
    #print('downsampling shape before add', x3.shape)
    
    res= Add()([x1, x3])
    #print('downsampling shape after add', res.shape)
    return res


# In[11]:


def res_upsampling(downsampling_block, previous_block, nfilter, k1, k2, k3, actvfc, p, add): 
    
    #print('downsampling_block.shape ', downsampling_block.shape)
    #print('previous_block.shape ', previous_block.shape)
    
    previous_block=UpSampling3D(size=(2,2,2))(previous_block)
    previous_block=Conv3D(filters=nfilter, kernel_size=(2,2,2), padding='same', activation=actvfc, 
                          kernel_initializer='he_uniform')(previous_block)
    
    if add:
        x=Add()([previous_block, downsampling_block])
    else: 
        x=Concatenate()([previous_block, downsampling_block])
    
    x= SpatialDropout3D(p)(x)    
    x1=Conv3D(filters=nfilter, kernel_size=(k1,k1,k1), padding='same', kernel_initializer='he_uniform')(x)
    x1=BatchNormalization()(x1)
    x1= Activation(actvfc)(x1)
    
    x2=Conv3D(filters=nfilter, kernel_size=(k2,k2,k2), padding='same', kernel_initializer='he_uniform')(x1) 
    x2=BatchNormalization()(x2)
    x2= Activation(actvfc)(x2)
    
    x3=Conv3D(filters=nfilter, kernel_size=(k3,k3,k3), padding='same', kernel_initializer='he_uniform')(x2) 
    x3=BatchNormalization()(x3)
    x3= Activation(actvfc)(x3)
    #print('upsampling shape before add', x3.shape)
    res= Add()([x1, x3])
    #print('upsampling shape before add', res.shape)
    return res


# In[16]:


def get_3DAda(h=512, w=512,p=0.5,k1=3,k2=3, k3=3, nfilter=32,actvfc='relu', blocks=9, slices=128, channels=2, add=True):
    # Input_shape=(height, width slices, channels)
    inp=Input((h, w, slices,channels))
    first_block=frst_blck(inp, nfilter, k1, k2, k3, actvfc)
    
    if blocks==3:
        
        down1=res_downsampling(first_block, nfilter*2, k1, k2, k3, actvfc, p) 
        output= last_blck(first_block, down1, nfilter, k1, k2, k3, actvfc, p, add)
         
        
    if blocks==5: 
        down1=res_downsampling(first_block, nfilter*2, k1, k2, k3, actvfc, p) 
        down2=res_downsampling(down1, nfilter*4, k1, k2, k3, actvfc, p) 
        up3=res_upsampling(down1, down2, nfilter*2, k1, k2, k3, actvfc, p, add) 
        output= last_blck(first_block, up3, nfilter, k1, k2, k3, actvfc, p, add)
        
    if blocks==7:
        down1=res_downsampling(first_block, nfilter*2, k1, k2, k3, actvfc, p) 
        down2=res_downsampling(down1, nfilter*4, k1, k2, k3, actvfc, p) 
        down3=res_downsampling(down2, nfilter*8, k1, k2, k3, actvfc, p) 
        up4=res_upsampling(down2, down3, nfilter*4, k1, k2, k3, actvfc, p, add) 
        up5=res_upsampling(down1, up4, nfilter*2, k1, k2, k3, actvfc, p, add) 
        output= last_blck(first_block, up5, nfilter, k1, k2, k3, actvfc, p, add)
        
    if blocks==9:
        down1=res_downsampling(first_block, nfilter*2, k1, k2, k3, actvfc, p) 
        down2=res_downsampling(down1, nfilter*4, k1, k2, k3, actvfc, p) 
        down3=res_downsampling(down2, nfilter*8, k1, k2, k3, actvfc, p) 
        down4=res_downsampling(down3, nfilter*16, k1, k2, k3, actvfc, p)
        up5=res_upsampling(down3, down4, nfilter*8, k1, k2, k3, actvfc, p, add) 
        up6=res_upsampling(down2, up5, nfilter*4, k1, k2, k3, actvfc, p, add) 
        up7=res_upsampling(down1, up6, nfilter*2, k1, k2, k3, actvfc, p, add) 
        output= last_blck(first_block, up7, nfilter, k1, k2, k3, actvfc, p, add)
    
    model= Model(inputs=inp, outputs=output)
    return model


# In[46]:


def prediction(kmodel, crpimg): 
    imarr=np.array(crpimg).astype(np.float32)   
    imarr = np.expand_dims(imarr, axis=0) #Adds a new dimension in the 0 axis that is the batch
    
    return kmodel.predict(imarr)


