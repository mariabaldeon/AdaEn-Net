import numpy as np
import pandas as pd
import keras
import logging
from keras import optimizers
from keras.models import model_from_json
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from keras import backend as K
from .AdaBN_3D import get_3DAda, prediction
from math import sqrt
from numpy import linalg as LA
from keras.utils import to_categorical
import math
import timeit
import matplotlib.pyplot as plt
from .ImageGenerator_3d import ImageDataGenerator


class MOEAD3D(object):
    def __init__(self, parameters, X_train, X_test, y_train, y_test):
        self.n_hyper = 9
        self.pop_size = parameters["pop_size"]
        self.nei_size = parameters["nei_size"]
        self.max_iter = parameters["max_gen"]
        self.penalty = parameters["penalty"]
        self.batch_size = parameters["batch_size"]
        self.w_tloss = parameters["alpha"]
        self.w_eloss = parameters["beta"]
        self.n_epochs = parameters["epochs"]
        self.ref_per = 0.8
        self.X_train_r = X_train
        self.X_val_r = X_test
        self.y_train_r = y_train
        self.y_val_r = y_test

        # =========================================================================================================
        # Hyperparameter Tuning
        # p= Spatial Dropout probability uniform [0-0.7]
        # ker_sizei= kernel size for the ith convolutional layer in the Residual framework. i=1,2,3 size=[1,3,5]
        # num_filters= number of the initial filters. The filters will be doubled in the downsampling and halved in the upsampling size=[4,8,16,32,64]
        # act_func= activation function throughout the architecture ['relu', 'elu']
        # blocks= Total number of blocks downsampling+upsampling [3, 5, 7, 9]
        # merge= type of merging layer in long connections [1-Add(), 0-Concatenation()]
        # alpha= learning rate [10^-6, 10^-2]
        # n_epochs= number of epochs to train per model
        # n_hyper= number of hyperparameters to be changes
        # Genotype=[p,ker_size1,ker_size2,ker_size3, num_filters,act_func,alpha, blocks, merge]
        # =========================================================================================================
        self.alpha = [1e-03, 9e-04, 8e-04, 7e-04, 6e-04, 5e-04, 4e-04, 3e-04, 2e-04, 1e-04,
                      9e-05, 8e-05, 7e-05, 6e-05, 5e-05, 4e-05, 3e-05, 2e-05, 1e-05]
        self.num_filters = [32, 16, 8]
        self.p = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        self.ker_size1 = [1, 3, 5, 7]
        self.ker_size2 = [1, 3, 5, 7]
        self.ker_size3 = [1, 3, 5, 7]
        self.act_func = ['relu', 'elu']
        self.blocks = [3, 5, 7, 9]
        self.merge = [0, 1]

    # Generates a random genotype
    def gen_genotype(self):
        pi = self.p[np.random.randint(0, len(self.p))]
        ker_size1i = self.ker_size1[np.random.randint(0, len(self.ker_size1))]
        ker_size2i = self.ker_size2[np.random.randint(0, len(self.ker_size2))]
        ker_size3i = self.ker_size3[np.random.randint(0, len(self.ker_size3))]
        num_filtersi = self.num_filters[np.random.randint(0, len(self.num_filters))]
        act_funci = self.act_func[np.random.randint(0, len(self.act_func))]
        alphai = self.alpha[np.random.randint(0, len(self.alpha))]
        blocksi = self.blocks[np.random.randint(0, len(self.blocks))]
        mergei = self.merge[np.random.randint(0, len(self.merge))]
        gene = [pi, ker_size1i, ker_size2i, ker_size3i, num_filtersi, act_funci, alphai, blocksi, mergei]
        return gene

        # Generates the names of the logging files

    def log_name(self, generation):
        location = "SearchLogs3D"
        if not os.path.exists(location):
            os.makedirs(location)
        logger = []
        weights_name = []
        params = []
        for i in range(0, self.pop_size):
            name = location + "/" + str(generation) + '_' + str(i) + '_training.log'
            w_name = location + "/" + str(generation) + '_' + str(i) + '_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
            p_name = location + "/" + str(generation) + '_' + str(i) + '_res+alpha.log'
            logger.append(name)
            weights_name.append(w_name)
            params.append(p_name)
        return logger, weights_name, params

        # Generates weight vectors for the whole population/pop_size (lambda in MOEA).
        # As in the paper, the weight is element of i/H for H=pop_size and i from 0 to H. Formula of H is specific for 2 OF.

    def generate_weight_vectors(self, pop_size):
        H = self.pop_size - 1

        # weight_vectors=the weights generated for the objective functions
        weight_vectors = []
        for i in range(self.pop_size):
            w_z1 = (i / H)
            w_z2 = 1 - w_z1
            weight_vectors.append((w_z1, w_z2))
        return weight_vectors

    # Generates the matrix neighbors of an individual based on the weights of the individual and the weights of the rest of the population
    # Compute the distance of any two vectors and generate the neighborhood
    # In the matrix, the row number determines the index of the individual in the weight_vectors and the numbers
    # in the row the index of the neighboring lambdas in the weight_vectors
    def compute_neighbors(self, weight_vectors, nei_size):

        # neighbor=matrix with the index of the neighbors
        neighbor = np.zeros((len(weight_vectors), self.nei_size))

        # for each weight
        for j in range(len(weight_vectors)):
            # Select the weight of the individual-> tuple with the weight for each OF
            weight = weight_vectors[j]
            distance = pd.DataFrame(columns=['i', 'dist'])

            # Compute the distance to other weights
            for i in range(len(weight_vectors)):
                # Compare the initial weight with all other weights
                # if i!=j:
                weight_2 = weight_vectors[i]
                dist = sqrt((weight_2[0] - weight[0]) ** 2 + (weight_2[1] - weight[1]) ** 2)
                # Save the distance with all weights
                distance = distance.append({'i': i, 'dist': dist}, ignore_index=True)

            # Sort the index by distance
            distance = distance.sort_values(by=['dist'])
            dist_m = distance.as_matrix()
            # Add only the top neighbor size index
            neighbor[j, :] = dist_m[:self.nei_size, 0]

        return neighbor

    # Randonmly generates one parent from the individual's neighbor
    def generate_parents(self, nei_size, neighbor, individual_index, OF):

        j = individual_index
        # Randomly generate one index of the parent from the neighbor of the individual
        parents = np.random.choice(neighbor[j], 1, replace=False)

        # Generates another parent if it is the same as the individual
        while parents == individual_index:
            parents = np.random.choice(neighbor[j], 1, replace=False)

        # Save the information in dataframes
        parent1 = OF.loc[
            OF['Var'] == 'x_0_' + str(int(parents)), ['p', 'ker_size1', 'ker_size2', 'ker_size3', 'n_filters',
                                                      'act_func', 'alpha', 'blocks', 'merge']]
        individual = OF.loc[OF['Var'] == 'x_0_' + str(j), ['p', 'ker_size1', 'ker_size2', 'ker_size3', 'n_filters',
                                                           'act_func', 'alpha', 'blocks', 'merge']]

        return parent1, individual

    # Generate the child's hyperparameters through recombination of the individual and the parents
    def recombination(self, parent1, individual, p):
        # child_hyper=child's hyperparameters
        child_hyper = []

        # Recombination. With p probability we select either of the parent's genotype
        for k in range(self.n_hyper):
            # Generate the probability
            prob = np.random.uniform(0, 1)
            if prob < self.p:
                child_hyper.append(parent1.iloc[0, k])
            if prob >= self.p:
                child_hyper.append(individual.iloc[0, k])

        return child_hyper

    # Computes the mutation probabilitaty for each gene.
    # Value reduces with more generations
    # gen is the number of generation
    def mutation_prob(self, gen):
        fi_0 = min(20 / self.n_hyper, 1)
        p_n = max(fi_0 * (1 - (math.log(gen - 1 + 1) / (math.log(self.max_iter)))), 1 / self.n_hyper)
        return p_n

    # Does Mutation. With p probability we do a mutation to each gene.
    def mutation(self, p, child_hyper):

        # Generate a new gene that will replace certain values of the child_hyper
        mutation_gene = self.gen_genotype()

        # for each gene
        for i in range(len(child_hyper)):
            # compute mutation probability
            prob = np.random.uniform(0, 1)

            if prob <= self.p:
                # Make sure you are really changing the gene
                while child_hyper[i] == mutation_gene[i]:
                    mutation_gene = self.gen_genotype()

                child_hyper[i] = mutation_gene[i]
        return child_hyper

    # Calculates the BI cost function of the neighbor and the new child
    def calculate_BI(self, OF_neighbor, Z_ref, max_OF, child_OF, nei_weights):
        # Obtains the OF of the neighbor
        OF_nei = np.matrix([[OF_neighbor.iloc[0, 1]], [OF_neighbor.iloc[0, 2]]])

        # Retrieves the OF of the minimum point. Normalize only architechture size
        Zm_ref = np.matrix(
            [[(Z_ref[1] - Z_ref[1]) / (max_OF[1] - Z_ref[1])], [(Z_ref[2] - Z_ref[2]) / (max_OF[2] - Z_ref[2])]])

        # Retrieves the OF of the new individual
        OF_chi = np.matrix([[child_OF[1]], [child_OF[2]]])

        # Compute the BI OF
        d1_nei = LA.norm((OF_nei - Zm_ref).T * nei_weights) / LA.norm(nei_weights)
        d2_nei = LA.norm(OF_nei - (Zm_ref + d1_nei * (nei_weights / LA.norm(nei_weights))))
        ObjFunc_nei = d1_nei + self.penalty * d2_nei

        # Compute the BI OF of the child
        d1_chi = LA.norm((OF_chi - Zm_ref).T * nei_weights) / LA.norm(nei_weights)
        d2_chi = LA.norm(OF_chi - (Zm_ref + d1_chi * (nei_weights / LA.norm(nei_weights))))
        ObjFunc_chi = d1_chi + self.penalty * d2_chi

        return ObjFunc_nei, ObjFunc_chi

    # Computes the real pareto points from all the population checked
    def real_pareto(self, models_checked):
        models_checked1 = models_checked.copy()
        models_checked1 = models_checked1.sort_values(['total_loss', 'param_count'])

        # Create new components with the OF normalized (Adaptive Normalization)
        models_checked1['train_loss_norm'] = models_checked1['train_loss']
        models_checked1['val_loss_norm'] = models_checked1['val_loss']
        models_checked1['total_loss_norm'] = (models_checked1['total_loss'] - Z_ref[1]) / (max_OF[1] - Z_ref[1])
        models_checked1['param_count_norm'] = (models_checked1['param_count'] - Z_ref[2]) / (max_OF[2] - Z_ref[2])

        s = 0
        OF3 = pd.DataFrame(columns=['Var', 'p', 'ker_size1', 'ker_size2', 'ker_size3', 'n_filters', 'act_func',
                                    'alpha', 'blocks', 'merge',
                                    'train_loss', 'val_loss', 'param_count', 'total_epochs', 'min_loss', 'total_loss',
                                    'train_loss_norm', 'val_loss_norm', 'total_loss_norm',
                                    'param_count_norm', 'real_var'])
        OF3 = OF3.append({'Var': 'x_0_' + str(s), 'p': models_checked1['p'].iloc[s],
                          'ker_size1': models_checked1['ker_size1'].iloc[s],
                          'ker_size2': models_checked1['ker_size2'].iloc[s],
                          'ker_size3': models_checked1['ker_size3'].iloc[s],
                          'n_filters': models_checked1['n_filters'].iloc[s],
                          'act_func': models_checked1['act_func'].iloc[s],
                          'alpha': models_checked1['alpha'].iloc[s], 'blocks': models_checked1['blocks'].iloc[s],
                          'merge': models_checked1['merge'].iloc[s],
                          'train_loss': models_checked1['train_loss'].iloc[s],
                          'val_loss': models_checked1['val_loss'].iloc[s],
                          'param_count': models_checked1['param_count'].iloc[s],
                          'total_epochs': models_checked1['total_epochs'].iloc[s],
                          'min_loss': models_checked1['min_loss'].iloc[s],
                          'total_loss': models_checked1['total_loss'].iloc[s],
                          'train_loss_norm': models_checked1['train_loss_norm'].iloc[s],
                          'val_loss_norm': models_checked1['val_loss_norm'].iloc[s],
                          'total_loss_norm': models_checked1['total_loss_norm'].iloc[s],
                          'param_count_norm': models_checked1['param_count_norm'].iloc[s],
                          'real_var': models_checked1['Var'].iloc[s]}, ignore_index=True)
        last_param_count = OF3['param_count_norm'].iloc[0]
        s = 1
        for i in range(1, len(models_checked1)):
            last_param_count = OF3['param_count_norm'].iloc[s - 1]
            new_param_count = models_checked1['param_count_norm'].iloc[i]
            if new_param_count < last_param_count:
                OF3 = OF3.append({'Var': 'x_0_' + str(s), 'p': models_checked1['p'].iloc[i],
                                  'ker_size1': models_checked1['ker_size1'].iloc[i],
                                  'ker_size2': models_checked1['ker_size2'].iloc[i],
                                  'ker_size3': models_checked1['ker_size3'].iloc[i],
                                  'n_filters': models_checked1['n_filters'].iloc[i],
                                  'act_func': models_checked1['act_func'].iloc[i],
                                  'alpha': models_checked1['alpha'].iloc[i],
                                  'blocks': models_checked1['blocks'].iloc[i],
                                  'merge': models_checked1['merge'].iloc[i],
                                  'train_loss': models_checked1['train_loss'].iloc[i],
                                  'val_loss': models_checked1['val_loss'].iloc[i],
                                  'param_count': models_checked1['param_count'].iloc[i],
                                  'total_epochs': models_checked1['total_epochs'].iloc[i],
                                  'min_loss': models_checked1['min_loss'].iloc[i],
                                  'total_loss': models_checked1['total_loss'].iloc[s],
                                  'train_loss_norm': models_checked1['train_loss_norm'].iloc[i],
                                  'val_loss_norm': models_checked1['val_loss_norm'].iloc[i],
                                  'total_loss_norm': models_checked1['total_loss_norm'].iloc[s],
                                  'param_count_norm': models_checked1['param_count_norm'].iloc[i],
                                  'real_var': models_checked1['Var'].iloc[i]}, ignore_index=True)
                s += 1
        return OF3

    # Total loss fc
    def total_loss_fc(self, train_loss, val_loss, min_loss):
        return self.w_tloss * train_loss + val_loss + self.w_eloss * ((self.n_epochs - min_loss) / self.n_epochs)

    # Initialize the training model in keras
    # loss coeficients
    smooth = 0.5
    threshold = 0

    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / ((K.sum(y_true_f * y_true_f)) + K.sum(y_pred_f * y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1. - self.dice_coef(y_true, y_pred)

    path = 'C:\\Users\\mariabaldeon\\Desktop\\Datasets\\Prostate MR Dataset\\3D\\V_Net\\128x128x32 (1mm,1mm,3mm)\\k4\\'
    # path='C:\\Users\\mariabaldeon\\Documents\\Research\\3D AdaResU-Net\\Prostate\\Dataset\\VNet\\'

    # Importing the pre processed data in the text file.
    self.X_train_r = np.load(path + "X4_trainVnt3mm.npy")
    self.X_val_r = np.load(path + "X4_testVnt3mm.npy")
    self.y_train_r = np.load(path + "y4_trainVnt3mm.npy")
    self.y_val_r = np.load(path + "y4_testVnt3mm.npy")

    # Save information of the images
    _, height, width, slices, channels = self.X_train_r.shape

    # Resize the input matrix so that it satisfies (batch, x, y, z,channels)
    print(self.X_train_r.shape)
    print(self.y_train_r.shape)
    print(self.X_val_r.shape)
    print(self.y_val_r.shape)

    # Normalize the data
    self.X_train_r = pre_processing(self.X_train_r)
    self.X_val_r = pre_processing(self.X_val_r)

    print(np.max(self.X_train_r), np.min(self.X_train_r))
    print(np.max(self.X_val_r), np.min(self.X_val_r))

    print(np.unique(self.y_train_r))
    print(np.unique(self.y_val_r))

    # Data Generator for the X and Y, includes data augmentation
    datagenX = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5,
                                  horizontal_flip=True, data_format='channels_last')
    datagenY = ImageDataGenerator(rotation_range=90, width_shift_range=0.4, height_shift_range=0.4, zoom_range=0.5,
                                  horizontal_flip=True, data_format='channels_last')

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_generator = datagenX.flow(self.X_train_r, self.batch_size = self.batch_size, seed = seed)
    mask_generator = datagenY.flow(self.y_train_r, self.batch_size = self.batch_size, seed = seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    # Genotype=[p,ker_size1,ker_size2,ker_size3, num_filters,act_func,alpha, blocks, merge]
    # creates the model based on the gene(genotype) and trains based on backpropgation
    # Saves all the information of the model in the logger files
    # The gene must be a list with length equal to the number of hyperparameters to tune.
    # It is expected that gene, logger, weights_name and params are for that particular individual
    def model_train_bp(self, generation, gene, logger, weights_name, params, indv, height, width, slices, channels):
        i = indv
        model = get_3DAda(h=height, w=width, self.p = gene[0], k1 = gene[1], k2 = gene[2], k3 = gene[3], nfilter = gene[
                                                                                                                       4], actvfc =
        gene[5],
        self.blocks = gene[7], slices = slices, channels = channels, add = gene[8])
        model.summary()
        self.alpha = gene[6]

        # Compile the model
        adam = optimizers.Adam(lr=self.alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=self.dice_coef_loss, optimizer=adam)

        # Stream epoch results to csv file
        csv_logger = CSVLogger(logger)
        model_check = ModelCheckpoint(filepath=weights_name, monitor='val_loss', verbose=0, save_best_only=True)
        early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, mode='auto')
        logging.basicConfig(filename=params, level=logging.INFO)
        logging.info(
            'generation: %s individual= %s p= %s k1= %s k2= %s k3= %s nfilter= %s act= %s alpha= %s blocks= %s add= %s',
            str(generation), str(i), str(gene[0]), str(gene[1]), str(gene[2]), str(gene[3]), str(gene[4]),
            str(gene[5]), str(gene[6]), str(gene[7]), str(gene[8]))

        # Fit the model
        history = model.fit_generator(train_generator, steps_per_epoch=(self.X_train_r.shape[0] / self.batch_size),
                                      validation_data=(self.X_val_r, self.y_val_r), epochs=self.n_epochs,
                                      callbacks=[csv_logger, model_check, early_stopper])

        if math.isnan(history.history['loss'][-1]):
            validation_loss = 1000
            train_loss = 1000
        elif math.isnan(history.history['val_loss'][-1]):
            validation_loss = 1000
            train_loss = 1000
        else:
            validation_loss = np.mean(history.history['val_loss'][-5:])
            train_loss = np.mean(history.history['loss'][-5:])
        train_parameters = model.count_params()
        min_index = np.argmin(history.history['val_loss'])
        total_epochs = len(history.history['val_loss'])

        del model
        K.clear_session()

        return train_loss, validation_loss, train_parameters, total_epochs, min_index

    # gene=[p,ker_size1,ker_size2,ker_size3, num_filters,act_func,alpha, blocks, merge]
    start_time_algo = timeit.default_timer()
    # 1.1 Initialize the algorithm
    # Randomly generate genotype for the initial population

    # List that saves all the generated genotypes
    genotype = []

    # Dataframe that saves all the information of each individual
    models_checked = pd.DataFrame(columns=['Var', 'p', 'ker_size1', 'ker_size2', 'ker_size3', 'n_filters',
                                           'act_func', 'alpha', 'blocks', 'merge', 'train_loss', 'val_loss',
                                           'param_count',
                                           'total_epochs', 'min_loss', 'total_loss'])

    # Generates the logger names for the whole population in an specific generation
    logger0, weights_name0, params0 = self.log_name(0, self.pop_size)

    # Creates the initial population
    for i in range(0, self.pop_size):
        gene = self.gen_genotype()
        print(gene)
        train_loss, validation_loss, train_params, total_epochs, min_loss = self.model_train_bp(0, gene, logger0[i],
                                                                                                weights_name0[i],
                                                                                                params0[i], i, height,
                                                                                                width, slices, channels)
        # Compute total loss
        total_loss = self.total_loss_fc(train_loss, validation_loss, min_loss)
        print(total_loss)
        genotype.append(gene)
        models_checked = models_checked.append({'Var': 'x_0_' + str(i), 'p': gene[0], 'ker_size1': gene[1],
                                                'ker_size2': gene[2], 'ker_size3': gene[3], 'n_filters': gene[4],
                                                'act_func': gene[5], 'alpha': gene[6], 'blocks': gene[7],
                                                'merge': gene[8], 'train_loss': train_loss,
                                                'val_loss': validation_loss, 'param_count': train_params,
                                                'total_epochs': total_epochs, 'min_loss': min_loss,
                                                'total_loss': total_loss}, ignore_index=True)

    # 1.2. Calculate the reference point
    model = get_3DAda(h=height, w=width, k1=int(np.min(self.ker_size1)), k2=int(np.min(self.ker_size2)),
                      k3=int(np.min(self.ker_size3)),
                      nfilter=int(np.min(self.num_filters)), self.blocks = int(
        np.min(self.blocks)), slices = slices, channels = channels )
    min_parameters = model.count_params()

    Z1_min = self.ref_per * np.min(models_checked['train_loss'])
    Z2_min = self.ref_per * np.min(models_checked['total_loss'])
    Z3_min = min_parameters

    Z_ref = [Z1_min, Z2_min, Z3_min]

    # index for evolution of population
    start = 1

    # 1.3 Generate the uniformly distributed weight vectors

    # Generate the weights of the objective functions for the whole population/pop_size weight
    weight_vectors = self.generate_weight_vectors(self.pop_size)

    # Determine the neighbors for each individual based on the distance of the weights
    neighbor = self.compute_neighbors(weight_vectors, self.nei_size)

    # STEP 2 Evolution!

    # j = individual i=generation
    OF = models_checked.copy()

    # Initialize or update the vector of maximum value of OF for Adaptive Normalization
    model = get_3DAda(h=height, w=width, k1=int(np.max(self.ker_size1)), k2=int(np.max(self.ker_size2)),
                      k3=int(np.max(self.ker_size3)),
                      nfilter=int(np.max(self.num_filters)), self.blocks = int(
        np.max(self.blocks)), slices = slices, channels = channels )
    max_parameters = model.count_params()

    # Dice coefficient of the validation+ Dice coefficient of the train*weight+ epoch with min loss
    max_total_loss = 2 + 1 * self.w_tloss
    max_OF = [1, max_total_loss, max_parameters]

    # Do all for all generations
    for i in range(start, self.max_iter):

        start_time = timeit.default_timer()

        # Create new components with the OF normalized (Adaptive Normalization)
        OF['train_loss_norm'] = OF['train_loss']
        OF['val_loss_norm'] = OF['val_loss']
        OF['total_loss_norm'] = (OF['total_loss'] - Z_ref[1]) / (max_OF[1] - Z_ref[1])
        OF['param_count_norm'] = (OF['param_count'] - Z_ref[2]) / (max_OF[2] - Z_ref[2])

        # Generates the logger names for the whole population in an specific generation
        logger, weights_name, params = self.log_name(generation=i, self.pop_size = self.pop_size)

        # Compute the probability of mutation. Since to compute the probability you assume we start in generation 2
        prob = self.mutation_prob(i + 1)

        # Do for all individuals j in generation i
        for j in range(self.pop_size):

            # Randomly select one index from the neighborhood of j and generate a new candidate solution "child_hyper"
            # from the parent and the individual
            parent1, individual = self.generate_parents(self.nei_size, neighbor, j, OF)

            # Generate the child's hyperparameters. With 1/2 prob we select either the parents genotype or the individuals genotype.
            child_hyper = self.recombination(parent1, individual, 1 / 2)

            # Mutation.
            child_hyper = self.mutation(prob, child_hyper)

            # Assures the same models are not trained
            while child_hyper in genotype:
                # Mutation
                child_hyper = self.mutation(prob, child_hyper)
            print('child_hyper:', child_hyper)

            # Train the child
            train_loss, validation_loss, train_params, total_epochs, min_loss = self.model_train_bp(i, child_hyper,
                                                                                                    logger[j],
                                                                                                    weights_name[j],
                                                                                                    params[j], j,
                                                                                                    height, width,
                                                                                                    slices, channels)
            # Compute total loss
            total_loss = self.total_loss_fc(train_loss, validation_loss, min_loss)
            print(total_loss)

            models_checked = models_checked.append(
                {'Var': 'x_' + str(i) + '_' + str(j), 'p': child_hyper[0], 'ker_size1': child_hyper[1],
                 'ker_size2': child_hyper[2], 'ker_size3': child_hyper[3],
                 'n_filters': child_hyper[4], 'act_func': child_hyper[5], 'alpha': child_hyper[6],
                 'blocks': child_hyper[7], 'merge': child_hyper[8], 'train_loss': train_loss,
                 'val_loss': validation_loss, 'param_count': train_params,
                 'total_epochs': total_epochs, 'min_loss': min_loss,
                 'total_loss': total_loss}, ignore_index=True)
            genotype.append(child_hyper)

            # Adaptive Normalization
            child_OF = [train_loss, (total_loss - Z_ref[1]) / (max_OF[1] - Z_ref[1]),
                        (train_params - Z_ref[2]) / (max_OF[2] - Z_ref[2])]

            # Calculate BI OF of each neighbor and compare with the child
            for m in range(self.nei_size):
                # m=neighbor
                # select the index of the neighbor
                neighbor_child = int(neighbor[j][m])
                nei_weights = np.asarray(weight_vectors[neighbor_child]).reshape((2, 1))

                # Retrieves the OF values of the neighbor
                OF_neighbor = OF.loc[OF['Var'] == 'x_0_' + str(neighbor_child), ['train_loss_norm', 'total_loss_norm',
                                                                                 'param_count_norm']]

                # Calculates the BI OF value for the neighbor m
                ObjFunc_nei, ObjFunc_chi = self.calculate_BI(OF_neighbor, Z_ref, max_OF, child_OF, nei_weights)

                # If the maximum cost of the new child is less than the neighbor, replace as the new optimal OF
                # If the cost of the new child is less than the neighbor, replace as the new optimal OF
                if ObjFunc_chi <= ObjFunc_nei:
                    # Eliminate the OF with lesser value than the child
                    OF = OF[OF["Var"] != 'x_0_' + str(neighbor_child)]
                    # Add the new pareto non optimal solution
                    OF = OF.append(
                        {'Var': 'x_0_' + str(neighbor_child), 'p': child_hyper[0], 'ker_size1': child_hyper[1],
                         'ker_size2': child_hyper[2], 'ker_size3': child_hyper[3], 'n_filters': child_hyper[4],
                         'act_func': child_hyper[5], 'alpha': child_hyper[6], 'blocks': child_hyper[7],
                         'merge': child_hyper[8], 'train_loss': train_loss, 'val_loss': validation_loss,
                         'param_count': train_params, 'total_epochs': total_epochs, 'min_loss': min_loss,
                         'total_loss': total_loss, 'train_loss_norm': child_OF[0], 'val_loss_norm': validation_loss,
                         'total_loss_norm': child_OF[1], 'param_count_norm': child_OF[2]}, ignore_index=True)
                    OF = OF.sort_values(by=['Var'])

            # Update the reference point
            if self.ref_per * train_loss < Z_ref[0]: Z_ref[0] = self.ref_per * train_loss
            if self.ref_per * total_loss < Z_ref[1]: Z_ref[1] = self.ref_per * total_loss

            OF.to_csv('OF.csv')
            models_checked.to_csv('models_checked.csv')
            pareto_solutions = self.real_pareto(models_checked)
            pareto_solutions.to_csv('pareto_solutions.csv')
        elapsed = timeit.default_timer() - start_time
        logging.info('generation time: %s', str(elapsed))

    # Save info into csv file
    final_time = timeit.default_timer() - start_time_algo
    OF.to_csv('OF.csv')
    models_checked.to_csv('models_checked.csv')
    logging.info('Time Elapsed: %s', str(final_time))
    pareto_solutions = self.real_pareto(models_checked)
    pareto_solutions.to_csv('pareto_solutions.csv')