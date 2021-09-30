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
import timeit
import os
import re
import math
import SimpleITK as sitk
#2D Model 
from AdaEn2D.AdaBN_2D import get_2DAda, prediction
from AdaEn2D.MOEAD2D import dice_coef, dice_coef_loss
#3D Model 
from AdaEn3D.AdaBN_3D import get_3DAda, prediction


# In[ ]:

class ModelEvaluate(object): 
    def __init__(self,parameters, patch_size, pix_spacing, X_val, y_val):
        self.gene3D=parameters["gene3D"]
        self.gene2D=parameters["gene2D"]
        self.stride=parameters["stride"]
        self.weights2D=parameters["weights2D"]
        self.weights3D=parameters["weights3D"]
        self.patch_size=patch_size
        self.pix_spacing=pix_spacing
        self.X_test3D=X_val
        self.y_test=y_val
        test_pat=self.y_test.shape[0]
        self.x=[self.pix_spacing[0]]*test_pat
        self.y=[self.pix_spacing[1]]*test_pat
        self.z=[self.pix_spacing[2]]*test_pat
    
    # Computes dice, jaccard, falsenegative error, false positive and hausdorff dist
    # gt= recieves one itk image gt or ground truth of shape (size,size, slice)
    # img=recieves one itk image segmentated of shape (size,size, slice)
    def compute_overlap_measures(self,gt, img):
        overlap_measure_filter=sitk.LabelOverlapMeasuresImageFilter()
        hausdorff_distance_filter=sitk.HausdorffDistanceImageFilter()
        overlap_measure_filter.Execute(gt,img)
        dice=overlap_measure_filter.GetDiceCoefficient()
        jaccard=overlap_measure_filter.GetJaccardCoefficient()
        false_negative=overlap_measure_filter.GetFalseNegativeError()
        false_positive=overlap_measure_filter.GetFalsePositiveError()
        hausdorff_distance_filter.Execute(gt,img)
        haus=hausdorff_distance_filter.GetHausdorffDistance()
        return dice, jaccard, false_negative, false_positive, haus
    
    # Returns the mean surface distance, median surface distance,
    def compute_surface_dist(self, gt, img):
        reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(gt, squaredDistance=False,
                                                                       useImageSpacing=True))
        reference_surface = sitk.LabelContour(gt)
        statistics_image_filter = sitk.StatisticsImageFilter()
        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())
        segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(img, squaredDistance=False,
                                                                      useImageSpacing=True))
        segmented_surface = sitk.LabelContour(img)

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

    def num_patches(self,img_dim, patch_dim, stride):    
        n_patch=math.trunc(img_dim/stride)
        if img_dim%stride==0:
            total_patches=n_patch
            lst_idx=(n_patch-1)*stride
            end_patch=lst_idx+patch_dim
            padding=end_patch-img_dim
            return total_patches, padding
        lst_idx=n_patch*stride
        end_patch=lst_idx+patch_dim
        padding=end_patch-img_dim
        total_patches=n_patch+1
        return total_patches, padding
    
    #Returns the matix with the predicted images shape (# images, pixel height, pixel width, slices)
    #X: input matrix with images shape (# images, row, col, slices, ch)
    #model: model from keras

    def prediction_matrix2D(self,X, model):
        num, row,col,sl, ch=X.shape
        y_pred_matrix=np.zeros((X.shape))
        time=np.zeros((num, 1))

        for i in range(num):
            Xi=np.zeros((sl, row, col, ch))
            # Reshape as input for 2D preprocessing
            for j in range(sl):
                Xi[j,:,:,:]=X[i,:,:,j,:]

            start_time = timeit.default_timer()
            y_pred=model.predict(x=Xi, batch_size=64) 
            elapsed = timeit.default_timer() - start_time
            time[i,:]=elapsed
            # Save the 3D image prediction
            for k in range(sl):
                y_pred_matrix[i,:,:,k,:]=y_pred[k,:,:,:]

        return y_pred_matrix, time

    def prediction_matrix_crop(self,X, model):
        num, row,col,sl, ch=X.shape
        pt_row, pt_col, pt_sl=self.patch_size
        str_row, str_col, str_sl=self.stride

        num_row, pad_row=self.num_patches(row, pt_row, str_row)
        num_col, pad_col=self.num_patches(col, pt_col, str_col)
        num_sl, pad_sl=self.num_patches(sl, pt_sl, str_sl)
        X_pad=np.zeros((num, row+pad_row, col+pad_col, sl+pad_sl, ch))
        X_pad[:, pad_row:, pad_col:, pad_sl:,:]=X
        y_pred_matrix=np.zeros(X_pad.shape)

        V=np.zeros(X_pad.shape)
        for i in range(num):
            for j in range(num_row):
                for k in range(num_col):
                    for m in range(num_sl):
                        row_in=j*str_row
                        col_in=k*str_col
                        sl_in=m*str_sl
                        row_fin=row_in+pt_row
                        col_fin=col_in+pt_col
                        sl_fin=sl_in+pt_sl
                        Xi=X_pad[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]
                        yi=prediction(model, Xi) 
                        print("yi", np.max(yi), np.min(yi))
                        y_pred_matrix[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=y_pred_matrix[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]+yi
                        Vi=np.zeros(X_pad.shape)
                        Vi[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=1.
                        V=V+Vi
        #compute the average of the predictions
        y_pred_matrix=np.true_divide(y_pred_matrix, V)
        print("y_pred_matrix", np.max(y_pred_matrix), np.min(y_pred_matrix))
        y_pred_matrix=y_pred_matrix[:, pad_row:, pad_col:, pad_sl:,:]
        return y_pred_matrix

    def connected_component(self, y_pred):
        num, r, c, s, ch= y_pred.shape
        y_new=np.zeros(y_pred.shape)
        for i in range(num):
            yi=y_pred[i,:,:,:,0]
            yi=sitk.GetImageFromArray(yi)

            # Apply threshold
            thfilter=sitk.BinaryThresholdImageFilter()
            thfilter.SetInsideValue(1)
            thfilter.SetOutsideValue(0)
            thfilter.SetLowerThreshold(0.5)
            yi = thfilter.Execute(yi)
            cc = sitk.ConnectedComponentImageFilter()
            yi = cc.Execute(sitk.Cast(yi,sitk.sitkUInt8))
            arrCC=np.transpose(sitk.GetArrayFromImage(yi).astype(dtype=float), [1, 2, 0])
            lab=np.zeros(int(np.max(arrCC)+1),dtype=float)

            for j in range(1,int(np.max(arrCC)+1)):
                lab[j]=np.sum(arrCC==j)
            activeLab=np.argmax(lab)
            yi = (yi==activeLab)
            yi=sitk.GetArrayFromImage(yi).astype(dtype=float)
            y_new[i,:,:,:,0]=yi
        return y_new
    
    def eval_metrics_dist(self,y_true, y_pred, set_spacing):
        thres=0.5
        tpy="uint8"
        num, h, w, s, c=y_true.shape

        y_true=y_true.reshape(y_true.shape[:-1])
        y_true=y_true.astype(dtype=tpy)
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
                img.SetSpacing((self.x[i], self.y[i], self.z[i]))
                gt.SetSpacing((self.x[i], self.y[i], self.z[i]))

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

            # If there is segmentation 
            # Compute the sensitivity, dice coeff, jaccard and surface distance measurements
            if (len(np.unique(y_truei))==2 and len(np.unique(y_predi))==2):
                dice, jaccard, false_negative, false_positive, haus=self.compute_overlap_measures(gt, img)
                mean_surface_distance,median_surface_distance, std_surface_distance, per95_surface_distance=self.compute_surface_dist(gt, img)
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

            metrics_cnn=pd.DataFrame(metric_coef_ind, columns=["val_dice", "val_recall","val_jaccard","val_false_negative_error", "val_false_positive_error", "val_haussdorf_dist","val_mean_surface_distance", "val_median_surface_distance","val_std_surface_distance", "val_95_haussdorf_dist"])
        return metrics_cnn

    
    def prediction_model_3D(self):
        model3D= get_3DAda(h=self.patch_size[0], w=self.patch_size[1] , p=self.gene3D[0],k1=self.gene3D[1],
                    k2=self.gene3D[2], k3=self.gene3D[3], nfilter=self.gene3D[4],actvfc=self.gene3D[5], 
                           blocks=self.gene3D[7], slices=self.patch_size[2], channels=self.channels,
                         add=self.gene3D[8] )
        model3D.load_weights(self.weights3D)

        #Verify the loaded model is correct
        model3D.summary()

        #Compile the model
        adam=optimizers.Adam(lr=self.gene3D[6], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model3D.compile(loss=dice_coef_loss, optimizer=adam, metrics=['accuracy'])

        y_test_pred3D=self.prediction_matrix_crop(self.X_test3D, model3D)
        print("red 3D", y_test_pred3D)

        # Apply Connected Component analysis
        y_test_pred3Dcc=self.connected_component(y_test_pred3D)
        print("red 3D cc", y_test_pred3Dcc)

        return y_test_pred3D, y_test_pred3Dcc


    def prediction_model_2D(self):
        model2D= get_2DAda(h=self.height,w=self.width, p=self.gene2D[0],k1=self.gene2D[1],k2=self.gene2D[2], 
                    k3=self.gene2D[3], nfilter=self.gene2D[4], actvfc=self.gene2D[5],
                       blocks=self.gene2D[7], channels=self.channels, add=self.gene2D[8])
        model2D.load_weights(self.weights2D)

        #Verify the loaded model is correct
        model2D.summary()

        #Compile the model
        adam=optimizers.Adam(lr=self.gene2D[6], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model2D.compile(loss=dice_coef_loss, optimizer=adam, metrics=['accuracy'])

        y_test_pred2D, test_time=self.prediction_matrix2D(self.X_test3D, model2D)

        # Apply Connected Component analysis
        y_test_pred2Dcc=self.connected_component(y_test_pred2D)

        return y_test_pred2D, y_test_pred2Dcc
    
    def evaluate(self):
        # Training logs 
        location="EvalLogs"
        if not os.path.exists(location):
            os.makedirs(location)
            
        _, self.height, self.width, _, self.channels=self.X_test3D.shape
        
        print(np.max(self.X_test3D),np.min(self.X_test3D))
        print(np.unique(self.y_test))

        y_test_pred3D, y_test_pred3Dcc=self.prediction_model_3D()
        print("y_test_pred3D", y_test_pred3D)
        print("y_test_pred3Dcc", y_test_pred3Dcc)
        y_test_pred2D, y_test_pred2Dcc=self.prediction_model_2D()
        metrics_test2Dcc =self.eval_metrics_dist(self.y_test, y_test_pred2Dcc, True)
        metrics_test2Dcc =metrics_test2Dcc[["val_dice", "val_recall","val_mean_surface_distance", "val_95_haussdorf_dist"]]
        metrics_test2Dcc.to_csv(location+"/metrics_test2Dcc.csv")
        print(metrics_test2Dcc)

        metrics_test3Dcc =self.eval_metrics_dist(self.y_test, y_test_pred3Dcc, True)
        metrics_test3Dcc =metrics_test3Dcc[["val_dice", "val_recall","val_mean_surface_distance", "val_95_haussdorf_dist"]]
        metrics_test3Dcc.to_csv(location+"/metrics_test3Dcc.csv")
        print(metrics_test3Dcc)

        y_test_pred=np.divide(y_test_pred3D+y_test_pred2D, 2.)
        y_test_predcc=self.connected_component(y_test_pred)

        metrics_test =self.eval_metrics_dist(self.y_test, y_test_predcc, True)
        metrics_test =metrics_test[["val_dice", "val_recall","val_mean_surface_distance", "val_95_haussdorf_dist"]]
        metrics_test.to_csv(location+"/metrics_test.csv")

