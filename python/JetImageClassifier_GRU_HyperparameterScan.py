import setGPU
import sys
import h5py
import glob
import numpy as np

# keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Input, GRU, Dropout, Flatten
from keras.layers import Concatenate, Reshape, BatchNormalization
from keras.layers import MaxPooling2D, MaxPooling3D
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.regularizers import l1
from generatorGRU import DataGenerator
import random

# hyperparameters
import GPy, GPyOpt

####################################################

from keras.activations import relu, selu, elu
# myModel class
class myModel():
    def __init__(self, input_train_files, input_test_files, optmizer_index=0, GRU_units=300,
                 DNN_neurons=40, 
                 DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):
        self.input_test_files = input_test_files
        self.input_train_files = input_train_files
        self.activation = [relu, selu, elu]
        self.optimizer = ['adam', 'nadam','adadelta']
        self.optimizer_index = optmizer_index
        self.GRU_units = GRU_units
        self.DNN_neurons = DNN_neurons
        self.DNN_layers = DNN_layers
        self.DNN_activation_index = DNN_activation_index
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.__x_test = np.array([])
        self.__y_test = np.array([])
        for i in range(min(5, len(input_test_files))):
            f =  h5py.File(input_test_files[i],"r")
            X = np.array(f.get('jetConstituentList'))            
            y = np.array(f.get('jets')[0:,-6:-1])
            self.__x_test = np.concatenate([self.__x_test, X], axis = 0) if self.__x_test.size else X
            self.__y_test = np.concatenate([self.__y_test, y], axis = 0) if self.__y_test.size else y
        print(self.__x_test.shape, self.__y_test.shape)
        self.__model = self.build()

    #  model
    def build(self):        
        inputArray = Input(shape=(self.__x_test.shape[1], self.__x_test.shape[2]))
        x = GRU(self.GRU_units, activation="tanh",
                recurrent_activation='hard_sigmoid', name='gru')(inputArray)
        x = Dropout(self.dropout)(x)
        ####
        for i in range(0,self.DNN_layers):
            x = Dense(self.DNN_neurons, activation=self.activation[self.DNN_activation_index], 
                      kernel_initializer='lecun_uniform', name='dense_%i' %i)(x)
            x = Dropout(self.dropout)(x)
        #
        output = Dense(5, activation='softmax', kernel_initializer='lecun_uniform', 
                       name = 'output_softmax')(x)
        ####
        model = Model(inputs=inputArray, outputs=output)
        model.compile(optimizer=self.optimizer[self.optimizer_index], 
                      loss='categorical_crossentropy', metrics=['acc'])
        return model

    
    # fit model
    def model_fit(self):
        my_batch_per_file = int(10000/self.batch_size)
        myTrainGen = DataGenerator("TRAINING", self.input_train_files, self.batch_size, my_batch_per_file)
        myValGen = DataGenerator("VALIDATION", self.input_test_files, self.batch_size, my_batch_per_file)

        self.__model.fit_generator(generator=myTrainGen, epochs=n_epochs,
                                   steps_per_epoch= my_batch_per_file*len(self.input_train_files), validation_data = myValGen,
                                   validation_steps =  my_batch_per_file*len(self.input_test_files), verbose=0,
                                   callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                                                           ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0),
                                                           TerminateOnNaN()])
    # evaluate  model
    def model_evaluate(self):
        self.model_fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, 
                                           batch_size=self.batch_size, verbose=0)
        return evaluation


####################################################

# Runner function for model
# function to run  class

def run_model(inputTESTfiles, inputVALfiles, optmizer_index=0, GRU_units=300,
                 DNN_neurons=40, DNN_layers=2, DNN_activation_index=0, 
              dropout=0.2, batch_size=100, epochs=50):
    
    _model = myModel(inputTESTfiles, inputVALfiles, optmizer_index, 
                     GRU_units, DNN_neurons, DNN_layers, 
                     DNN_activation_index, dropout, batch_size, epochs)
    model_evaluation = _model.model_evaluate()
    return model_evaluation

####################################################


import glob
inputTrainFiles = glob.glob("/data/ML/mpierini/hls-fml/jetImage*_%sp*.h5" %sys.argv[1])
inputTestFiles = glob.glob("/data/ML/mpierini/hls-fml/VALIDATION/jetImage*_%sp*.h5" %sys.argv[1])
#inputTrainFiles = glob.glob("/data/ml/mpierini/hls-fml/jetImage*_%sp*.h5" %sys.argv[1])
#inputTestFiles = glob.glob("/data/ml/mpierini/hls-fml/VALIDATION/jetImage*_%sp*.h5" %sys.argv[1])
random.shuffle(inputTrainFiles)
random.shuffle(inputTestFiles)

n_epochs = 50
# Bayesian Optimization

# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'GRU_units',           'type': 'discrete',   'domain': (50, 100, 200, 300, 400, 500)},
          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)},
          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1, 2, 3, 4)},
          {'name': 'DNN_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'dropout',               'type': 'continuous', 'domain': (0.1, 0.4)},
          {'name': 'batch_size',            'type': 'discrete',   'domain': (50, 100, 200, 500)}]

# function to optimize model
def f(x):
    print(x)
    evaluation = run_model(inputTrainFiles, inputTestFiles,
                           optmizer_index = int(x[:,0]), 
                           GRU_units = int(x[:,1]), 
                           DNN_neurons = int(x[:,2]), 
                           DNN_layers = int(x[:,3]),
                           DNN_activation_index = int(x[:,4]),
                           dropout = float(x[:,5]),
                           batch_size = int(x[:,6]),
                           epochs = n_epochs)
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]

# run optimization
opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
opt_model.run_optimization(max_iter=10000)

print("x:",opt_model.x_opt)
print("y:",opt_model.fx_opt)

# print optimized model
print("""
Optimized Parameters:
\t{0}:\t{1}
\t{2}:\t{3}
\t{4}:\t{5}
\t{6}:\t{7}
\t{8}:\t{9}
\t{10}:\t{11}
\t{12}:\t{13}
""".format(bounds[0]["name"],opt_model.x_opt[0],
           bounds[1]["name"],opt_model.x_opt[1],
           bounds[2]["name"],opt_model.x_opt[2],
           bounds[3]["name"],opt_model.x_opt[3],
           bounds[4]["name"],opt_model.x_opt[4],
           bounds[5]["name"],opt_model.x_opt[5],
           bounds[6]["name"],opt_model.x_opt[6]))
print("optimized loss: {0}".format(opt_model.fx_opt))

print(opt_model.x_opt)
