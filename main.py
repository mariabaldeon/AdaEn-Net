#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
from Datasets.Promise12.DataManager import DataManager
from AdaEn2D.MOEAD2D import MOEAD2D
from Train.train2D import ModelTrain2D
#from Evaluation.evaluation_model import ModelEvaluate

params=dict()
params["DataManager"]=dict()
params["SearchParams"]=dict()
params["TrainParams"]=dict()
params["EvalParams"]=dict()

############ Set Image Preprocessing Parameters ###############################
# Voxel spacing
params["DataManager"]["VolSpa"]=np.asarray([1,1,1.5],dtype=float)
# Fixed size of the image
params["DataManager"]["VolSize"]=np.asarray([128,128,64],dtype=int)
# Total Number of images in the dataset
params["DataManager"]["NumImages"]=50
# Number of validation images
params["DataManager"]["TestImages"]=10
# Path to the dataset
basePath=os.getcwd()
pathset = os.path.join(basePath, "/home/mgbaldeon/Research/EMONAS/Github/Datasets/Promise12/Images")
params["DataManager"]["dirTrain"]=pathset

############ Set Search Parameters ###############################
# 2D architecture
# Training Epochs for candidate architectures
params["SearchParams"]["epochs"]=120
# Population size
params["SearchParams"]["pop_size"]=8
# neighborhood size
params["SearchParams"]["nei_size"]=4
# Number of generations
params["SearchParams"]["max_gen"]=40
# penalty
params["SearchParams"]["penalty"]=0.01
# batch size to train candidate architectures
params["SearchParams"]["batch_size"]=10
# Alpha parameter in expected segmentation error loss function
params["SearchParams"]["alpha"]=0.25
# Beta parameter in expected segmentation error loss function
params["SearchParams"]["beta"]=0.25

############ Set Training Parameters ###############################
# 2D architecture
# Genotype to decode into AdaEn-Net architecture
# Genotype=[p, k1, k2, k3, nfilter, activation fuction, learning rate, number of blocks, merge function]
params["TrainParams"]["gene"]=[0.15,1,3,7,16,'relu',0.000005,7,0]
# Training Epochs
params["TrainParams"]["num_epochs"]=1000
# batch size to train architectures
params["TrainParams"]["batch_size"]=40

############ Set Evaluation Parameters ###############################
params["EvalParams"]["gene"]=[0.003,32,7,1,0,2, "convP3d_3x3", "conv3d_3x3x3", "conv2d_5x5", "conv3d_1x1x1"]
# stride to make the prediction
params["EvalParams"]["stride"]=(128,128,1)
# Path to the weights you want to evaluate
params["EvalParams"]["path"]="weights"

if sys.argv[1]=="-search2D":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Architecture Search for 2D AdaEn-Net
    NAS2D=MOEAD2D(params["SearchParams"], DM.X_train, DM.X_test, DM.y_train, DM.y_test)
    NAS2D.search()

elif sys.argv[1]=="-train2D":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Train
    model=ModelTrain2D(params["TrainParams"], DM.X_train, DM.X_test, DM.y_train, DM.y_test)
    model.train()

elif sys.argv[1]=="-evaluate":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Evaluate
    EV=ModelEvaluate(params["EvalParams"],params["TrainParams"]["patch_size"],
                params["DataManager"]["VolSpa"], DM.X_test, DM.y_test)
    EV.evaluate()
