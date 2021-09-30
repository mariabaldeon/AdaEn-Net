
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import sys
from Datasets.Promise12.DataManager import DataManager
from AdaEn2D.MOEAD2D import MOEAD2D
from AdaEn3D.MOEAD3D import MOEAD3D
from Train.train2D import ModelTrain2D
from Train.train3D import ModelTrain3D
from Ensemble.EnsembleEvaluation import ModelEvaluate

params=dict()
params["DataManager"]=dict()
params["SearchParams2D"]=dict()
params["SearchParams3D"]=dict()
params["TrainParams2D"]=dict()
params["TrainParams3D"]=dict()
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
params["SearchParams2D"]["epochs"]=120
# Population size
params["SearchParams2D"]["pop_size"]=8
# neighborhood size
params["SearchParams2D"]["nei_size"]=4
# Number of generations
params["SearchParams2D"]["max_gen"]=40
# penalty
params["SearchParams2D"]["penalty"]=0.01
# batch size to train candidate architectures
params["SearchParams2D"]["batch_size"]=10
# Alpha parameter in expected segmentation error loss function
params["SearchParams2D"]["alpha"]=0.25
# Beta parameter in expected segmentation error loss function
params["SearchParams2D"]["beta"]=0.25

# 3D architecture
params["SearchParams3D"]["epochs"]=120
params["SearchParams3D"]["pop_size"]=8
params["SearchParams3D"]["nei_size"]=4
params["SearchParams3D"]["max_gen"]=40
params["SearchParams3D"]["penalty"]=0.01
params["SearchParams3D"]["batch_size"]=1
params["SearchParams3D"]["alpha"]=0.25
params["SearchParams3D"]["beta"]=0.25
# patch size to train architectures
params["SearchParams3D"]["patch_size"]=(96,96,16)

############ Set Training Parameters ###############################
# 2D architecture
# Genotype to decode into AdaEn-Net architecture
# Genotype=[p, k1, k2, k3, nfilter, activation fuction, learning rate, number of blocks, merge function]
params["TrainParams2D"]["gene"]=[0.15,1,3,7,16,'relu',0.0004,7,0]
# Training Epochs
params["TrainParams2D"]["num_epochs"]=3000
# batch size to train architectures
params["TrainParams2D"]["batch_size"]=40

# 3D architecture
params["TrainParams3D"]["gene"]=[0,3,1,5,32,'elu',0.00005, 5, 0]
# Training Epochs
params["TrainParams3D"]["num_epochs"]=6000
# batch size to train architectures
params["TrainParams3D"]["batch_size"]=4
# patch size to train architectures
params["TrainParams3D"]["patch_size"]=(96,96,16)

############ Set Evaluation Parameters ###############################
params["EvalParams"]["gene2D"]=[0.15,1,3,7,16,'relu',0.0004,7,0]
params["EvalParams"]["gene3D"]=[0,3,1,5,32,'elu',0.00005, 5, 0]
# stride to make the prediction with the 3D FCN
params["EvalParams"]["stride"]=(48,48,8)
# Path to the weights you want to evaluate
params["EvalParams"]["weights2D"]='weights/weights_k1_2D.hdf5'
params["EvalParams"]["weights3D"]='weights/weights_k1_3D.hdf5'

if sys.argv[1]=="-search2D":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Architecture Search for 2D AdaEn-Net
    NAS2D=MOEAD2D(params["SearchParams2D"], DM.X_train, DM.X_test, DM.y_train, DM.y_test)
    NAS2D.search()

if sys.argv[1]=="-search3D":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Architecture Search for 3D AdaEn-Net
    NAS3D=MOEAD3D(params["SearchParams3D"], DM.X_train, DM.X_test, DM.y_train, DM.y_test)
    NAS3D.search()

elif sys.argv[1]=="-train2D":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Train
    model=ModelTrain2D(params["TrainParams2D"], DM.X_train, DM.X_test, DM.y_train, DM.y_test)
    model.train()

elif sys.argv[1]=="-train3D":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Train
    model=ModelTrain3D(params["TrainParams3D"], DM.X_train, DM.X_test, DM.y_train, DM.y_test)
    model.train()

elif sys.argv[1]=="-evaluate":
    # Preprocessing images
    DM=DataManager(params["DataManager"])
    DM.preprocess()

    # Evaluate
    EV=ModelEvaluate(params["EvalParams"],params["TrainParams3D"]["patch_size"],
                params["DataManager"]["VolSpa"], DM.X_test, DM.y_test)
    EV.evaluate()