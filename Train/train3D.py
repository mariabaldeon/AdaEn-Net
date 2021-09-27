
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import keras
import logging
from keras import optimizers
from keras.models import model_from_json
from keras.callbacks import CSVLogger,ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN, LearningRateScheduler, TensorBoard, Callback  
from keras import backend as K
from AdaBN_3D import get_3DAda, prediction
from preprocessing import del_out_3D, norm_max_3D
import math
import timeit
from ImageGenerator_3dcrop import ImageDataGenerator


# In[2]:


#gene=p, k1, k2, k3, nfilter, activation fuction, learning rate
#x_23_5
gene=[[0,3,1,5,32,'elu',0.000005, 5, 0]]
weight_name=['0_weights.2423-0.89.hdf5']

#Define variables new_size: size of the image, iterations=number of models to train
img_size=128
patch_size=(96,96,16)
iterations=int(len(gene))
num_epochs=3000
add_weights=True
batch_size=4




def pre_processing(X): 
    #Eliminates pixels out of the 3 standard deviation
    X=del_out_3D(X, 3)
    #Normalize by dividing with the maximum value
    X=norm_max_3D(X)
    
    return X

# Initialize the training model in keras
eps=1e-7
#loss coeficients
smooth=0.5
threshold=0

def dice_coef(y_true, y_pred):
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
    #total number of patches that will be in the image in that dimension
    total_patch=math.ceil(img_dim/patch_dim)
    print('total_patch', total_patch)
    if total_patch==1: 
        return img_dim, total_patch
    # number of pixels lacking if would fit perfectly all the patches
    pix_dif=(patch_dim*total_patch)-img_dim
    print('pix_dif', pix_dif)
    #Divide evenly the lacking pixels in all the patches stride
    stride_dif=math.ceil(pix_dif/(total_patch-1))
    print('strid_dif', stride_dif)
    stride=patch_dim-stride_dif
    return stride, total_patch
# Convert the X_val matrix of X_val.shape(num, rows, columns, slices, channels) to a matrix of patches
# of X_val_patch.shape(num*patches_img, patch_row, patch_column, patch_slice, channels)
def val_convert_patch(X_val, patch_dim):
    num, row, col, sl, ch= X_val.shape
    pt_row, pt_col, pt_sl= patch_dim
    # compute the stride and num of patches per each dimension
    row_str, num_row=val_stride(row, pt_row)
    col_str, num_col=val_stride(col, pt_col)
    sl_str, num_sl=val_stride(sl, pt_sl)
    #print('row_str, num_row',row_str, num_row )
    #print('col_str, num_col',col_str, num_col )
    #print('sl_str, num_sl',sl_str, num_sl )
    
    # New X_val_patch matrix
    img_patch=num_row*num_col*num_sl
    total_patch=num*img_patch
    X_val_patch=np.zeros((total_patch, pt_row, pt_col, pt_sl, ch))
    #Populate the new matrix
    ix_patch=0
    #For each image
    for i in range(num):
        #For each row
        for j in range(num_row):
            #For each column
            for k in range(num_col): 
                #For each slice
                for m in range(num_sl): 
                    row_in=j*row_str
                    col_in=k*col_str
                    sl_in=m*sl_str
                    row_fin=row_in+pt_row
                    col_fin=col_in+pt_col
                    sl_fin=sl_in+pt_sl
                    #print('row_in, row_fin',row_in, row_fin )
                    #print('col_in, col_fin',col_in, col_fin )
                    #print('sl_in, sl_fin',sl_in, sl_fin )
                    X_val_patch[ix_patch,:,:,:,0]=X_val[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,0]
                    ix_patch=ix_patch+1
    return X_val_patch




path='/home/m/mariabaldeon/Documents/3D AdaResU-Net/Prostate/Dataset/Ensemble Testing/3D/K4/'

# Importing the pre processed data in the text file. 
X_train_r= np.load(path+"X4_trainEn.npy")
X_val_r= np.load(path+"X4_valEn.npy")
y_train_r= np.load(path+"y4_trainEn.npy")
y_val_r= np.load(path+"y4_valEn.npy")

# Save information of the images
_, height, width, slices, channels=X_train_r.shape

# Normalize the data
X_train_r= pre_processing(X_train_r)
X_val_r= pre_processing(X_val_r)

# Crop the validation data according to patch
X_val_r=val_convert_patch(X_val_r, patch_size )
y_val_r=val_convert_patch(y_val_r, patch_size )

print(X_train_r.shape)
print(y_train_r.shape)
print(X_val_r.shape)
print(y_val_r.shape)

print(np.max(X_train_r),np.min(X_train_r))
print(np.max(X_val_r),np.min(X_val_r))

print(np.unique(y_train_r))
print(np.unique(y_val_r))


# In[8]:


#Data Generator for the X and Y, includes data augmentation
datagenX = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5, horizontal_flip=True, data_format='channels_last', random_crop=patch_size)
datagenY = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5, horizontal_flip=True, data_format='channels_last', random_crop=patch_size)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

image_generator = datagenX.flow(X_train_r, batch_size=batch_size, seed=seed)
mask_generator = datagenY.flow(y_train_r, batch_size=batch_size, seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)


# In[ ]:


# Model Training 
logger=[]
weights_name=[]
params=[]
for i in range(0,iterations): 
    name=str(i)+'_training.log'
    w_name=str(i)+'_weights.{epoch:02d}-{val_dice_coef:.2f}.hdf5'
    p_name=str(i)+'_res+alpha.log'
    logger.append(name)
    weights_name.append(w_name)
    params.append(p_name)


# In[ ]:


for i in range(0,iterations): 

    #Start Timer
    start_time = timeit.default_timer()
    
    model= get_3DAda(h=patch_size[0],w=patch_size[1], p=gene[i][0],k1=gene[i][1],k2=gene[i][2], k3=gene[i][3], nfilter=gene[i][4],actvfc=gene[i][5], 
                   blocks=gene[i][7], slices=patch_size[2], channels=channels, add=gene[i][8])
    alpha=gene[i][6]
    
    model.summary()
    
    if add_weights: 
        model.load_weights(weight_name[i])

    #Compile the model
    adam=optimizers.Adam(lr=alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=dice_coef_loss, optimizer=adam, metrics=['accuracy',
                                                                dice_coef, 
                                                                recall])

    #Stream epoch results to csv file 
    csv_logger = CSVLogger(logger[i])
    model_check=ModelCheckpoint(filepath= weights_name[i] , monitor='val_loss', verbose=0, save_best_only=True)
    #early_stopper=EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, mode='auto')
    logging.basicConfig(filename=params[i], level=logging.INFO)
   
    #Fit the model
    history=model.fit_generator(train_generator, steps_per_epoch=(X_train_r.shape[0]/batch_size), 
                            validation_data=(X_val_r, y_val_r), epochs=num_epochs, 
                            callbacks=[csv_logger, model_check])
    
    #Save information of the best validation dice
    max_index=np.argmax(history.history['val_dice_coef'])
    max_dice_val=history.history['val_dice_coef'][max_index]
    dice_train=history.history['dice_coef'][max_index]
    logging.info('model= %s p= %s k1= %s k2= %s k3= %s nfilter= %s act= %s alpha= %s max dice val= %s train dice= %s epoch= %s ', 
                 str(i), str(gene[i][0]), str(gene[i][1]), str(gene[i][2]), str(gene[i][3]), str(gene[i][4]),
                 str(gene[i][5]), str(gene[i][6]), str(max_dice_val), str(dice_train), str(max_index))
    
    # Save elapsed time
    elapsed = timeit.default_timer() - start_time
    logging.info('Time Elapsed: %s', str(elapsed))

    #Save model to disk
    model_json=model.to_json()
    model_name=str(i)+'_model.jason'
    model_weights=str(i)+'_model.h5'
    with open(model_name,"w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_weights)
    print("Saved model to disk")
    
    # Delete info to release GPU memory
    del model 
    K.clear_session()

