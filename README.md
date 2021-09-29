# AdaEn-Net
In this work, we present AdaEn-Net, a self-adaptive 2D–3D ensemble of Fully Convolutional Netowrks (FCNs) for 3D medical image segmentation that incorporates volumetric data and adapts to a particular dataset by optimizing both the model’s performance and size. The AdaEn-Net consists of a 2D FCN that extracts intra-slice information and a 3D FCN that exploits inter-slice information. The architecture and hyperparameters of the 2D and 3D architectures are found through a multiobjective evolutionary based algorithm that maximizes the expected segmentation accuracy and minimizes the number of parameters in the network.

![alt text](https://github.com/mariabaldeon/AdaEn-Net/blob/main/Images/1-s2.0-S0893608020300848-gr3.jpg)

# Requirements
* Python 3.7
* Numpy 1.19.2
* Keras 2.3.1
* Tensorflow 1.14.0
* lhsmdu 1.1
* Pygmo 2.16.1
* Simpleitk 2.0.2

# Dataset
The prostate MR images from the PROMISE12 challenge is available [here](https://promise12.grand-challenge.org/). Firts, **you must download the dataset, put all images in a folder named *Images* and locate it in */Datasets/Promise12* for the code to run** (the path set by default for the image dataset is *Datasets/Promise12/Images)*.
The parameters used to preprocess the data are located in ```main.py``` in the ```params["DataManager"]``` dictionary. If you want to change any parameter, please do it here. 

# Architecture search 
The architecture search is done for the 2D FCN and 3D FCN. To carry out the 2D FCN architecture search run:
```
nohup python3 main.py -search2D & 
```
To carry out the 3D FCN architecture search run:
```
nohup python3 main.py -search3D & 
```

The output will for both will be a: (1) a .csv file named *pareto_solutions.csv* that contains all the solutions that approximate the Pareto Front. (2) *SearchLogs* folder with the logs of the search 

* (1) In the *pareto_solutions.csv* file, each pareto solution is in a row.  The solution in the first row minimizes the expected segmentation error, and the solution in the last row minimizes the size of the network. Select the architecture that best satisfies your requirements. For our experiments, we select the solution that minimizes the expected segmentation error (1st row). For each solution, the csv file provides the optimized hyperparameters and training information: learning_rate= learning rate, node2_inp = input to node 2, node3_inp = input to node 3, node4_inp= input to node 4, ops1= convolutional operation for node 1, ops2= convolutional operation for node 2, ops3= convolutional operation for node 3, ops4= convolutional operation for node 4, num_cells= total number of encoder-decoder cells, num_filters= number of filters for the first cell, total_loss= expected segmentation error loss, val_loss= validation loss, train_loss= training loss, and param_count= number of trainable parameters in the architecture. Note the validation performance in this search is not the final performance of the architecture. We only train for a maximum of 120 epochs during the optimization process. You must fully train the architecture from sctrach (see the directions above to fully train) and the select the weights that minimizes the validation error. 
* (2) In the *SearchLogs* folder the training loss and validation loss for each architecture trained during the search will be saved, plus the time it took to run each generation and the whole search. 

Due to the stochastic nature of the search, each run will end with a different approximate Pareto Front. To obtain the best results your must run the search with different seeds and select the architecture that has the best validation performance after fully training it.  
Finally, the parameters used to perform the search are located in ```main.py``` in the ```params["SearchParams2D"]``` dictionary for the 2D FCN search and in the ```params["SearchParams3D"]``` dictionary for the 3D FCN search. They are set according to the paper, if you want to change any parameter, please do it here.  

# Train model
To fully train the 2D FCN architecture run:
```
nohup python3 main.py -train2D &  
```
To fully train the 3D FCN architecture run:
```
nohup python3 main.py -train3D &  
```

The parameters used to perform the training are located in ```main.py``` in the ```params["TrainParams2D"]``` dictionary for the 2D FCN architecture and ```params["TrainParams3D"]``` dictionary for the 3D FCN architecture . The genotype for the best architectures found in our paper is used for default in the parameter ```params["TrainParams2D"]["gene"]``` and ```params["TrainParams3D"]["gene"]```. Hence, if you run the code as it is, you will fully train the architecture found with our experiments. **If you want to train another architectures you must change the parameter assigned to ```params["TrainParams2D"]["gene"]``` and ```params["TrainParams3D"]["gene"]```** . Specifically, we encode an architecture using a list with the following format: Genotype=[learning_rate,num_filters,num_cells, node2_in, node3_in, node4_in, ops_node1, ops_node2, ops_node3, ops_node4]. These hyperparameters are the same as provided in the *pareto_solutions.csv* file after a search. Therefore, if you want to train an architecture according to your own search just copy the results from the *pareto_solutions.csv* file in the ```main.py``` file using the format provided before (ie:  ```params["TrainParams"]["gene"]=[learning_rate,num_filters,num_cells, node2_in, node3_in, node4_in, ops_node1, ops_node2, ops_node3, ops_node4]```).

The ouput will be saved in a *TrainLogs* folder. There two types of outputs (1) the weights saved during the training process where the name has the following format weights.{epoch}--{validation_dice_coeff}.hdf5 (the best weight is the one that has the highest validation_dice_coeff) and (2) logs with the loss, dice coefficent, accuracy, and recall on each training epoch and the training time.  

# Evaluate a model
To evaluate the 2D-3D ensemble, first you **must create a folder named *weights* and locate the weight for the 2D and 3D FCN you want to evaluate in this folder, and verify the parameter assigned to ```params["EvalParams"]["gene"]``` is the same parameter you assigned to ```params["TrainParams"]["gene"]```  when training the architecture**(explained below). Then run: 
```
nohup python3 main.py -evaluate &  
```
The parameters used to perform the evaluation are located in ```main.py``` in the ```params["EvalParams"]``` dictionary. By default, the genotype for the best architecture found in our paper is assigned to ```params["EvalParams"]"]["gene"]```. Hence, if you run the code as it is, you will evaluate the architecture found with our experiments. If you want to evaluate another architecture you must change this parameter. Specifically, we encode an architecture using a list with the following format: Genotype=[learning_rate,num_filters,num_cells, node2_in, node3_in, node4_in, ops_node1, ops_node2, ops_node3, ops_node4].    

The ouput will be saved in the *EvalLogs* folder. There will be two .csv files as output (1) *CompiledMetrics.csv* will provide the mean evaluation metrics for the patients in the evaluation set. The metrics provided are val_dice= mean validation dice coefficient, val_hauss= mean validation hausdorff distance, val_MSD= mean validation mean surface distance, val_recall= mean validation recall  (2) *MetricsVal.csv* will provide the evalution metrics for each patient in the validation set. 

# Citation
If you use this code in your research, please cite our paper.
```
@article{calisto2020adaen,
  title={AdaEn-Net: An ensemble of adaptive 2D--3D Fully Convolutional Networks for medical image segmentation},
  author={Calisto, Maria Baldeon and Lai-Yuen, Susana K},
  journal={Neural Networks},
  volume={126},
  pages={76--94},
  year={2020},
  publisher={Elsevier}
}
```
