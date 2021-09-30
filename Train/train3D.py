#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import keras
import logging
from keras import optimizers
from keras.models import model_from_json
from keras.callbacks import CSVLogger,ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from AdaEn3D.AdaBN_3D import get_3DAda, prediction
import math
import timeit
import os
from AdaEn3D.ImageGenerator_3dcrop import ImageDataGenerator


def dice_coef(y_true, y_pred):
    smooth=0.5
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return(2.*intersection+smooth)/((K.sum(y_true_f*y_true_f)) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def recall(y_true, y_pred):
    y_true_f= K.flatten(y_true)
    y_pred_f= K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# Computes the stride for transforming the initial validation into a matrix of batches
# The stride is computed trying to fit the least number of patches that cover the whole image
#Returns the stride and total number of patches in the images
def val_stride(img_dim, patch_dim):
    total_patch=math.ceil(img_dim/patch_dim)
    if total_patch==1: 
        return img_dim, total_patch
    pix_dif=(patch_dim*total_patch)-img_dim
    stride_dif=math.ceil(pix_dif/(total_patch-1))
    stride=patch_dim-stride_dif
    return stride, total_patch

# Convert the X_val matrix of X_val.shape(num, rows, columns, slices, channels) to a matrix of patches
# of X_val_patch.shape(num*patches_img, patch_row, patch_column, patch_slice, channels)
def val_convert_patch(X_val, patch_dim):
    num, row, col, sl, ch= X_val.shape
    pt_row, pt_col, pt_sl= patch_dim
    row_str, num_row=val_stride(row, pt_row)
    col_str, num_col=val_stride(col, pt_col)
    sl_str, num_sl=val_stride(sl, pt_sl)
    
    img_patch=num_row*num_col*num_sl
    total_patch=num*img_patch
    X_val_patch=np.zeros((total_patch, pt_row, pt_col, pt_sl, ch))
    ix_patch=0
    for i in range(num):
        for j in range(num_row):
            for k in range(num_col): 
                for m in range(num_sl): 
                    row_in=j*row_str
                    col_in=k*col_str
                    sl_in=m*sl_str
                    row_fin=row_in+pt_row
                    col_fin=col_in+pt_col
                    sl_fin=sl_in+pt_sl
                    X_val_patch[ix_patch,:,:,:,0]=X_val[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,0]
                    ix_patch=ix_patch+1
    return X_val_patch


class ModelTrain3D(object): 
    def __init__(self,parameters, X_train, X_val, y_train, y_val):
        self.gene=parameters["gene"]
        self.num_epochs=parameters["num_epochs"]
        self.batch_size=parameters["batch_size"]
        self.patch_size=parameters["patch_size"]
        self.X_train_r= X_train
        self.X_val_r= X_val
        self.y_train_r= y_train
        self.y_val_r=y_val
        
    def train(self):
        location="TrainLogs/Results3D"
        if not os.path.exists(location):
            os.makedirs(location)

        # Save information of the images
        _, height, width, slices, channels=self.X_train_r.shape
        
        # Crop the validation data according to patch
        self.X_val_r=val_convert_patch(self.X_val_r, self.patch_size )
        self.y_val_r=val_convert_patch(self.y_val_r, self.patch_size )

        print(self.X_train_r.shape)
        print(self.y_train_r.shape)
        print(self.X_val_r.shape)
        print(self.y_val_r.shape)

        print(np.max(self.X_train_r),np.min(self.X_train_r))
        print(np.max(self.X_val_r),np.min(self.X_val_r))

        print(np.unique(self.y_train_r))
        print(np.unique(self.y_val_r))


        #Data Generator for the X and Y, includes data augmentation
        datagenX = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5,
                                      horizontal_flip=True, data_format='channels_last', random_crop=self.patch_size)
        datagenY = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5, 
                                      horizontal_flip=True, data_format='channels_last', random_crop=self.patch_size)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1

        image_generator = datagenX.flow(self.X_train_r, batch_size=self.batch_size, seed=seed)
        mask_generator = datagenY.flow(self.y_train_r, batch_size=self.batch_size, seed=seed)

        # combine generators into one which yields image and masks
        train_generator = zip(image_generator, mask_generator)


        # Model Training 
        logger=location+'/training.log'
        weights_name=location+'/weights.{epoch:02d}-{val_dice_coef:.2f}.hdf5'

        # Train the model with the training dataset and using the validation dataset. 
        # Saving all information of the epochs for further use.
        #Start Timer
        start_time = timeit.default_timer()

        model= get_3DAda(h=self.patch_size[0],w=self.patch_size[1], p=self.gene[0],k1=self.gene[1],
                         k2=self.gene[2], k3=self.gene[3], nfilter=self.gene[4],actvfc=self.gene[5], 
                   blocks=self.gene[7], slices=self.patch_size[2], channels=channels, add=self.gene[8])
        model.summary()

        #Compile the model
        adam=optimizers.Adam(lr=self.gene[6], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=dice_coef_loss, optimizer=adam, metrics=['accuracy', dice_coef, recall])

        #Stream epoch results to csv file 
        csv_logger = CSVLogger(logger)
        model_check=ModelCheckpoint(filepath= weights_name, monitor='val_loss', verbose=0, save_best_only=True)

        #Fit the model
        history=model.fit_generator(train_generator, steps_per_epoch=(self.X_train_r.shape[0]/self.batch_size), 
                                    validation_data=(self.X_val_r, self.y_val_r),
                                    epochs=self.num_epochs, callbacks=[csv_logger, model_check])

        #Save information of the best validation dice
        max_index=np.argmax(history.history['val_dice_coef'])
        max_dice_val=history.history['val_dice_coef'][max_index]
        dice_train=history.history['dice_coef'][max_index]
        logging.info('p= %s k1= %s k2= %s k3= %s nfilter= %s act= %s alpha= %s max dice val= %s train dice= %s epoch= %s ', str(self.gene[0]), str(self.gene[1]), str(self.gene[2]), str(self.gene[3]), str(self.gene[4]), str(self.gene[5]), str(self.gene[6]), str(max_dice_val), str(dice_train), str(max_index))

        # Save elapsed time
        elapsed = timeit.default_timer() - start_time
        logging.info('Time Elapsed: %s', str(elapsed))

