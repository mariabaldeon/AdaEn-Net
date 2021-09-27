
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import keras
import logging
from keras import optimizers
from keras.models import model_from_json
from keras.callbacks import CSVLogger,ModelCheckpoint, ReduceLROnPlateau, EarlyStopping 
from keras import backend as K
import timeit
import os
import re
import math

#2D Model 
from AdaBN_2D import get_2DAda, prediction
from preprocessing import del_out, norm_max

#3D Model 
from AdaBN_3D import get_3DAda, prediction
from preprocessing import del_out_3D, norm_max_3D


# In[3]:


pix_spacing=(1,1,1.5)
test_patients=10

# pixel spacing
x=[pix_spacing[0]]*test_patients
y=[pix_spacing[1]]*test_patients
z=[pix_spacing[2]]*test_patients

#Define 3D model
#23_5 cropped
gene3D=[[0,3,1,5,32,'elu',0.00005, 5, 0]]
patch_size=(96,96,16)
stride=(48,48,8)

#Define 2D model
#38_6
gene2D=[[0.15,1,3,7,16,'relu',0.0004,7,0]]

#Weights########### CHANGE#################
weights2D='weights/weights_k2_2D.hdf5'
weights3D='weights/weights_k2_3D.hdf5'


# In[4]:


# Computes dice, jaccard, falsenegative error, false positive and hausdorff dist
# gt= recieves one itk image gt or ground truth of shape (size,size, slice)
# img=recieves one itk image segmentated of shape (size,size, slice)
def compute_overlap_measures(gt, img):
    overlap_measure_filter=sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter=sitk.HausdorffDistanceImageFilter()
    overlap_measure_filter.Execute(gt,img)
    dice=overlap_measure_filter.GetDiceCoefficient()
    jaccard=overlap_measure_filter.GetJaccardCoefficient()
    false_negative=overlap_measure_filter.GetFalseNegativeError()
    false_positive=overlap_measure_filter.GetFalsePositiveError()
    # volume_diff=overlap_measure_filter.GetVolumeSimilarity()
    hausdorff_distance_filter.Execute(gt,img)
    haus=hausdorff_distance_filter.GetHausdorffDistance()
    return dice, jaccard, false_negative, false_positive, haus

# Returns the mean surface distance, median surface distance, 
# maximum surface distance and std surface distance
# img must be a 3D itk image of the segmentation
# gt must be the 3D itk image of the ground truth
def compute_surface_dist(gt, img):
    #COMPUTE MEAN SURFACE DISTANCE
    # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside 
    # relationship, is irrelevant)
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(gt, squaredDistance=False, 
                                                                   useImageSpacing=True))
    reference_surface = sitk.LabelContour(gt)
    
    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of voxels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum()) 
    
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(img, squaredDistance=False, 
                                                                  useImageSpacing=True))
    segmented_surface = sitk.LabelContour(img)
        
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    
    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances +                         list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances +                         list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        
    all_surface_distances = seg2ref_distances + ref2seg_distances
    
    mean_surface_distance = np.mean(all_surface_distances)
    median_surface_distance = np.median(all_surface_distances)
    std_surface_distance = np.std(all_surface_distances)
    per95_surface_distance = np.percentile(all_surface_distances, 95)
    return mean_surface_distance,median_surface_distance, std_surface_distance, per95_surface_distance


# In[5]:


# Initialize the training model in keras

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

def num_patches(img_dim, patch_dim, stride): 
    n_patch=math.trunc(img_dim/stride)
    
    # If the dimension is perfectly divisible. No padding and perfectly computed number of patches
    if img_dim%stride==0: 
        total_patches=n_patch
        lst_idx=(n_patch-1)*stride
        end_patch=lst_idx+patch_dim
        padding=end_patch-img_dim
        print('n_patch', n_patch)
        print('lst_idx', lst_idx)
        print('end_patch', end_patch)
        print('padding', padding)
        
        return total_patches, padding
        
    lst_idx=n_patch*stride
    end_patch=lst_idx+patch_dim
    padding=end_patch-img_dim
    total_patches=n_patch+1

    print('n_patch', n_patch)
    print('lst_idx', lst_idx)
    print('end_patch', end_patch)
    print('padding', padding)
    return total_patches, padding


# In[6]:


#Returns the matix with the predicted images shape (# images, pixel height, pixel width, slices)
#A matrix with the time it took to predict each slice of images

#X: input matrix with images shape (# images, row, col, slices, ch)
#model: model from keras

def prediction_matrix2D(X, model):
    num, row,col,sl, ch=X.shape
    y_pred_matrix=np.zeros((X.shape))
    time=np.zeros((num, 1))
    
    for i in range(num): 
        # Saves the image of each patient
        Xi=np.zeros((sl, row, col, ch))
        
        # Reshape as input for 2D preprocessing
        for j in range(sl): 
            Xi[j,:,:,:]=X[i,:,:,j,:]
        
        start_time = timeit.default_timer()
        
        # prediction recieves a matrix of shape (num, row, col, channel)
        y_pred=model.predict(x=Xi, batch_size=64) # returns size=(num,row,col,ch)
        print("prediction shape ", y_pred.shape)
        
        elapsed = timeit.default_timer() - start_time
        time[i,:]=elapsed
        
        # Save the 3D image prediction
        for k in range(sl): 
            y_pred_matrix[i,:,:,k,:]=y_pred[k,:,:,:]
    
    return y_pred_matrix, time 


# In[7]:


def prediction_matrix_crop(X, model): 
    num, row,col,sl, ch=X.shape
    pt_row, pt_col, pt_sl=patch_size
    str_row, str_col, str_sl=stride
    
    # total patches in each dimension and the padding at each dimension to have all strides
    num_row, pad_row=num_patches(row, pt_row, str_row)
    num_col, pad_col=num_patches(col, pt_col, str_col)
    num_sl, pad_sl=num_patches(sl, pt_sl, str_sl)
    
    X_pad=np.zeros((num, row+pad_row, col+pad_col, sl+pad_sl, ch))
    X_pad[:, pad_row:, pad_col:, pad_sl:,:]=X
    print('X_pad.shape', X_pad.shape)
    
    y_pred_matrix=np.zeros(X_pad.shape)
    # Counts the number of times the patch goes through the image to compute the average
    V=np.zeros(X_pad.shape)
    
    # for each patient
    for i in range(num): 
        #For each row patch
        for j in range(num_row):
            #For each column patch
            for k in range(num_col): 
                #For each slice patch
                for m in range(num_sl): 
                    print('patient ', i)
                    row_in=j*str_row
                    col_in=k*str_col
                    sl_in=m*str_sl
                    row_fin=row_in+pt_row
                    col_fin=col_in+pt_col
                    sl_fin=sl_in+pt_sl
                    print('row_in, row_fin',row_in, row_fin )
                    print('col_in, col_fin',col_in, col_fin )
                    print('sl_in, sl_fin',sl_in, sl_fin )
                    
                    Xi=X_pad[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]
                    yi=prediction(model, Xi) #output size=(1,size,size,slices,1)
                    # Add previous predictions and the current prediction
                    y_pred_matrix[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=y_pred_matrix[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]+yi
                    
                    # Compute how many times a prediction has been dome to that pixel
                    Vi=np.zeros(X_pad.shape)
                    Vi[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=1.
                    V=V+Vi
                    print('V max predictions', np.max(V))
    #compute the average of the predictions
    y_pred_matrix=np.true_divide(y_pred_matrix, V)
    #take the padding out 
    y_pred_matrix=y_pred_matrix[:, pad_row:, pad_col:, pad_sl:,:]
    print('y_pred_matrix.shape', y_pred_matrix.shape)
    return y_pred_matrix


# In[8]:


def pre_processing3D(X): 
    #Eliminates pixels out of the 3 standard deviation
    X=del_out_3D(X, 3)

    #Normalize by dividing with the maximum value
    X=norm_max_3D(X)
    
    return X


# In[9]:


def pre_processing2D(X): 
    num, row, col, sl, ch=X.shape
    # Matrix with post processing operations
    Xnew=np.zeros(X.shape)
    
    for i in range(num): 
        # Saves the image of each patient
        Xi=np.zeros((sl, row, col, ch))
        
        # Reshape as input for 2D preprocessing
        for j in range(sl): 
            Xi[j,:,:,:]=X[i,:,:,j,:]
        
        #Eliminates pixels out of the 3 standard deviation (recieves a matrix of shape (num, row, col, channel))
        Xi=del_out(Xi, 3)
        #Normalize by dividing with the maximum value
        Xi=norm_max(Xi)
        
        # Reshape as input for 3D image
        for k in range(sl): 
            Xnew[i,:,:,k,:]=Xi[k,:,:,:]
    
    return Xnew


# In[10]:


#y_pred: prediction in probabilities shape (# images, height, width, slices, channels)
def connected_component(y_pred): 
    num, r, c, s, ch= y_pred.shape
    y_new=np.zeros(y_pred.shape)
    for i in range(num):
        print('image num: ', i)
        yi=y_pred[i,:,:,:,0]
        #print('initial yi: ', np.unique(yi))
        yi=sitk.GetImageFromArray(yi)
        
        # Apply threshold
        thfilter=sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        yi = thfilter.Execute(yi)
        #print('after threshold: ', np.unique(yi))
        
        #connected component analysis (better safe than sorry)
        # labels the objects in a binary image. Each distinct object is assigned a unique label
        # The final object labels start with 1 and are consecutive. 
        # ObjectCount holds the number of connected components
        cc = sitk.ConnectedComponentImageFilter()
        yi = cc.Execute(sitk.Cast(yi,sitk.sitkUInt8))
        #print('after CC: ', np.unique(yi))        


        # Turn into a numpy array after the connected component analysis
        arrCC=np.transpose(sitk.GetArrayFromImage(yi).astype(dtype=float), [1, 2, 0])

        # Array of the length of all possible components in the image
        # If only 1 component is found (segments only 1 area) the array is size 1x2
        lab=np.zeros(int(np.max(arrCC)+1),dtype=float)
        #print('num of components: ', np.max(arrCC))

        # For each label after the connected component analysis 
        for j in range(1,int(np.max(arrCC)+1)):
            # Add the number of pixels that have that label
            lab[j]=np.sum(arrCC==j)
        #print('num of pixels per component: ', lab)

        # The label that has the biggest number of segmented pixels
        activeLab=np.argmax(lab)
        #print('maximum label: ', activeLab)

        # Keep only pixels of the image that have the activeLab label (label with most number of pixels) 
        yi = (yi==activeLab)
        #print('after yi==activeLab: ', np.unique(yi))        

        yi=sitk.GetArrayFromImage(yi).astype(dtype=float)
        #print("yi.shape ", yi.shape)
        
        y_new[i,:,:,:,0]=yi
    return y_new

# Evaluates 11 metrics of overlap and distance measure
#y_true: ground truth in binary to_categorical function shape (# images, height, width, slices, channels)
#y_pred: prediction in probabilities shape (# images, height, width, slices, channels)
#x= list with the mm in x axis of each patient, y= list with mm in y axis of each patient,  
import SimpleITK as sitk

def eval_metrics_dist(y_true, y_pred, set_spacing, x,y,z):
    thres=0.5
    tpy="uint8"
    num, h, w, s, c=y_true.shape
    
    # y_true.shape= (#images, height, width, slices)
    y_true=y_true.reshape(y_true.shape[:-1])
    y_true=y_true.astype(dtype=tpy)
    
    # y_pred.shape (#images, height, width, slices)
    y_pred=np.where(y_pred>=thres,1,0)
    y_pred=y_pred.reshape(y_pred.shape[:-1])
    y_pred=y_pred.astype(dtype=tpy)

    metric_coef_ind=np.zeros((num,10))
    
    for i in range(num): 
        
        y_predi=y_pred[i,:,:,:]
        y_truei=y_true[i,:,:,:]
        
        # USE ITK package to compute other measures
        img=sitk.GetImageFromArray(y_predi)
        gt=sitk.GetImageFromArray(y_truei)
    
        if set_spacing: 
            img.SetSpacing((x[i], y[i], z[i]))
            gt.SetSpacing((x[i], y[i], z[i]))
    
        # If there is no segmentation in the groundtruth or segmentation
        # cannot compute the sensitivity, dice coeff, jaccard and surface distance measurements
        if (len(np.unique(y_truei))==1 or len(np.unique(y_predi))==1): 
            
            metric_coef_ind[i,0]=np.nan
            metric_coef_ind[i,1]=np.nan
            metric_coef_ind[i,2]=np.nan
            metric_coef_ind[i,3]=np.nan
            metric_coef_ind[i,4]=np.nan
            metric_coef_ind[i,5]=np.nan
            metric_coef_ind[i,6]=np.nan
            metric_coef_ind[i,7]=np.nan
            metric_coef_ind[i,8]=np.nan
            metric_coef_ind[i,9]=np.nan
        
        # If there is segmentation in the groundtruth
        # Compute the sensitivity, dice coeff, jaccard and surface distance measurements
        if (len(np.unique(y_truei))==2 and len(np.unique(y_predi))==2):
            dice, jaccard, false_negative, false_positive, haus=compute_overlap_measures(gt, img)
            mean_surface_distance,median_surface_distance, std_surface_distance, per95_surface_distance=compute_surface_dist(gt, img)
            metric_coef_ind[i,0]=dice
            metric_coef_ind[i,1]=1-false_positive
            metric_coef_ind[i,2]=jaccard
            metric_coef_ind[i,3]=false_negative
            metric_coef_ind[i,4]=false_positive
            metric_coef_ind[i,5]=haus
            metric_coef_ind[i,6]=mean_surface_distance
            metric_coef_ind[i,7]=median_surface_distance
            metric_coef_ind[i,8]=std_surface_distance
            metric_coef_ind[i,9]=per95_surface_distance
        
        metrics_cnn=pd.DataFrame(metric_coef_ind, columns=["dice", "recall","jaccard",
                                                      "false_negative_error", "false_positive_error", "haus",
                                                      "mean_surface_distance", "median_surface_distance",
                                                      "std_surface_distance", "95 haus"])
    return metrics_cnn

import math
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


# In[11]:


#Create a list with the paths to each file in the 3D testing (the input for 2D and 3D are 3D images)
path='C:\\Users\\mariabaldeon\\Desktop\\Datasets\\Prostate MR Dataset\\3D\\V_Net\\128x128x64 (1mm, 1mm, 1.5mm)\\k2\\'
X_test=np.load(path+"X2_testVnt15mm.npy")
y_test=np.load(path+'y2_testVnt15mm.npy')


num, height, width, slices, channels=X_test.shape

print(X_test.shape)
print(y_test.shape)


# In[12]:


X_test3D=pre_processing3D(X_test)
X_test2D=pre_processing2D(X_test)
print(np.max(X_test3D),np.min(X_test3D))
print(np.max(X_test2D),np.min(X_test2D))
print(np.unique(y_test))


# In[13]:


def prediction_model_3D(X_test3D):
    model3D= get_3DAda(h=patch_size[0], w=patch_size[1] , p=gene3D[0][0],k1=gene3D[0][1],k2=gene3D[0][2], k3=gene3D[0][3], 
                   nfilter=gene3D[0][4],actvfc=gene3D[0][5], blocks=gene3D[0][7], slices=patch_size[2], channels=channels, 
                     add=gene3D[0][8] )
    model3D.load_weights(weights3D)
    
    #Verify the loaded model is correct 
    model3D.summary()

    #Compile the model
    adam=optimizers.Adam(lr=gene3D[0][6], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model3D.compile(loss=dice_coef_loss, optimizer=adam, metrics=['accuracy'])
    
    y_test_pred3D=prediction_matrix_crop(X_test3D, model3D)

    # Apply Connected Component analysis 
    y_test_pred3Dcc=connected_component(y_test_pred3D)
    
    return y_test_pred3D, y_test_pred3Dcc


# In[14]:


def prediction_model_2D(X_test2D):
    model2D= get_2DAda(h=height,w=width, p=gene2D[0][0],k1=gene2D[0][1],k2=gene2D[0][2], k3=gene2D[0][3],
                     nfilter=gene2D[0][4], actvfc=gene2D[0][5], 
                   blocks=gene2D[0][7], channels=channels, add=gene2D[0][8])
    model2D.load_weights(weights2D)

    #Verify the loaded model is correct 
    model2D.summary()

    #Compile the model
    adam=optimizers.Adam(lr=gene2D[0][6], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model2D.compile(loss=dice_coef_loss, optimizer=adam, metrics=['accuracy'])
    
    y_test_pred2D, test_time=prediction_matrix2D(X_test2D, model2D)

    # Apply Connected Component analysis 
    y_test_pred2Dcc=connected_component(y_test_pred2D)
    
    return y_test_pred2D, y_test_pred2Dcc


# In[ ]:


y_test_pred3D, y_test_pred3Dcc=prediction_model_3D(X_test3D)
y_test_pred2D, y_test_pred2Dcc=prediction_model_2D(X_test2D)
metrics_test2Dcc =eval_metrics_dist(y_test, y_test_pred2Dcc, True, x,y,z)
metrics_test2Dcc.to_csv("metrics_test2Dcc.csv")
metrics_test3Dcc =eval_metrics_dist(y_test, y_test_pred3Dcc, True, x,y,z)
metrics_test3Dcc.to_csv("metrics_test3Dcc.csv")

y_test_pred=np.divide(y_test_pred3D+y_test_pred2D, 2.)
y_test_predcc=connected_component(y_test_pred)

metrics_test =eval_metrics_dist(y_test, y_test_predcc, True, x,y,z)
metrics_test.to_csv("metrics_test.csv")

y_test_predv2=np.divide(y_test_pred3Dcc+y_test_pred2Dcc, 2.)
y_test_predv2cc=connected_component(y_test_predv2)
    
metrics_testcc =eval_metrics_dist(y_test, y_test_predv2cc, True, x,y,z)
metrics_testcc.to_csv("metrics_testcc.csv")

