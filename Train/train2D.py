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
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from AdaEn2D.AdaBN_2D import get_2DAda, prediction
from AdaEn2D.MOEAD2D import dice_coef, dice_coef_loss, change_format
import timeit
import os


# In[ ]:


def recall(y_true, y_pred):
    y_true_f= K.flatten(y_true)
    y_pred_f= K.flatten(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

class ModelTrain2D(object): 
    def __init__(self,parameters, X_train, X_val, y_train, y_val):
        self.gene=parameters["gene"]
        self.num_epochs=parameters["num_epochs"]
        self.batch_size=parameters["batch_size"]
        self.X_train_r= X_train
        self.X_val_r= X_val
        self.y_train_r= y_train
        self.y_val_r=y_val
        
    def train(self):

        location="TrainLogs/Results2D"
        if not os.path.exists(location):
            os.makedirs(location)

        # Save information of the images
        self.X_train_r, self.y_train_r,self.X_val_r, self.y_val_r=change_format(self.X_train_r, self.y_train_r,
                                                    self.X_val_r, self.y_val_r)

        _, height, width, channels=self.X_train_r.shape

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
                                      horizontal_flip=True, data_format='channels_last')
        datagenY = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5, 
                                      horizontal_flip=True, data_format='channels_last')

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

        model= get_2DAda(h=height,w=width, p=self.gene[0],k1=self.gene[1],k2=self.gene[2], 
                         k3=self.gene[3], nfilter=self.gene[4],actvfc=self.gene[5], 
                       blocks=self.gene[7], channels=channels, add=self.gene[8])
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

