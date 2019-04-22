import setGPU
import sys
import h5py
import glob
import numpy as np

# keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from keras.layers import Concatenate, Reshape, BatchNormalization
from keras.layers import MaxPooling2D, MaxPooling3D
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.regularizers import l1

# hyperparameters
import GPy, GPyOpt

####################################################

from keras.activations import relu, selu, elu
# myModel class
class myModel():
    def __init__(self, x_train, x_test, y_train, y_test, optmizer_index=0, DNN_neurons=40, 
                 DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):
        self.activation = [relu, selu, elu]
        self.optimizer = ['adam', 'nadam','adadelta']
        self.optimizer_index = optmizer_index
        self.DNN_neurons = DNN_neurons
        self.DNN_layers = DNN_layers
        self.DNN_activation_index = DNN_activation_index
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.__x_train, self.__x_test, self.__y_train, self.__y_test = x_train, x_test, y_train, y_test
        self.__model = self.build()
    
    #  model
    def build(self):
        inputArray = Input(shape=(input_shape,))
        x = Dense(self.DNN_neurons, activation=self.activation[self.DNN_activation_index], 
                  kernel_initializer='lecun_uniform', name='dense_0')(inputArray)
        x = Dropout(self.dropout)(x)
        ####
        for i in range(1,self.DNN_layers):
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
        self.__model.fit(self.__x_train, self.__y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        verbose=0,
                        validation_data=[self.__x_test, self.__y_test],
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

def run_model(x_train, x_test, y_train, y_test, optmizer_index=0, DNN_neurons=40, 
              DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):
    
    _model = myModel(x_train, x_test, y_train, y_test, optmizer_index, 
                     DNN_neurons, DNN_layers, DNN_activation_index, 
                 dropout, batch_size, epochs)
    model_evaluation = _model.model_evaluate()
    return model_evaluation

####################################################


import glob
X = np.array([])
Y = np.array([])
for fileIN in glob.glob("/eos/project/d/dshep/hls-fml/NEWDATA/jetImage*_%sp*.h5" %sys.argv[1]):
    print(fileIN)
    f = h5py.File(fileIN, 'r')
    myFeatures = np.array(f.get('jets')[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]])
    myTarget = np.array(f.get('jets')[0:,-6:-1])
    X = np.concatenate([X,myFeatures], axis = 0) if X.size else myFeatures
    Y = np.concatenate([Y,myTarget], axis = 0) if Y.size else myTarget
    print(X.shape, Y.shape)
input_shape = len([12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

n_epochs = 50
# Bayesian Optimization

# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)},
          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1, 2, 3)},
          {'name': 'DNN_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'dropout',               'type': 'continuous', 'domain': (0.1, 0.4)},
          {'name': 'batch_size',            'type': 'discrete',   'domain': (50, 100, 200, 500)}]

# function to optimize model
def f(x):
    print(x)
    evaluation = run_model(X_train, X_test, Y_train, Y_test,
                           optmizer_index = int(x[:,0]), 
                           DNN_neurons = int(x[:,1]), 
                           DNN_layers = int(x[:,2]),
                           DNN_activation_index = int(x[:,3]),
                           dropout = float(x[:,4]),
                           batch_size = int(x[:,5]),
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
""".format(bounds[0]["name"],opt_model.x_opt[0],
           bounds[1]["name"],opt_model.x_opt[1],
           bounds[2]["name"],opt_model.x_opt[2],
           bounds[3]["name"],opt_model.x_opt[3],
           bounds[4]["name"],opt_model.x_opt[4],
           bounds[5]["name"],opt_model.x_opt[5]))
print("optimized loss: {0}".format(opt_model.fx_opt))

print(opt_model.x_opt)
