import setGPU
import sys
import h5py
import glob
import numpy as np

# keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from keras.layers import Concatenate, BatchNormalization, Activation
from keras.layers import MaxPooling2D, MaxPooling3D
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.regularizers import l1
from generatorCNN import DataGenerator

# hyperparameters
import GPy, GPyOpt

####################################################

# myModel class
class myModel():
    def __init__(self, input_test_files, input_val_files, optmizer_index=0, CNN_filters=10, 
                 CNN_filter_size=5, CNN_MaxPool_size=5, CNN_layers=2, CNN_activation_index=0, DNN_neurons=40, 
                 DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):  
        self.input_test_files = input_test_files
        self.input_val_files = input_val_files
        self.activation = ['relu', 'selu', 'elu']
        self.optimizer = ['adam', 'nadam','adadelta']
        self.optimizer_index = optmizer_index
        self.CNN_filters = CNN_filters
        self.CNN_filter_size = CNN_filter_size
        self.CNN_MaxPool_size = CNN_MaxPool_size
        self.CNN_layers = CNN_layers
        self.CNN_activation_index = CNN_activation_index
        self.DNN_neurons = DNN_neurons
        self.DNN_layers = DNN_layers
        self.DNN_activation_index = DNN_activation_index
        self.dropout = dropout
        self.batch_size = batch_size
        # here an epoch is a single file
        self.epochs = epochs
        self.__x_test = np.array([])
        self.__y_test = np.array([])
        for i in range(min(5, len(input_val_files))):
            f =  h5py.File(input_val_files[i],"r")
            X =  np.array(f.get('jetImage'))
            X =  X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
            y = np.array(f.get('jets')[0:,-6:-1])
            self.__x_test = np.concatenate([self.__x_test, X], axis = 0) if self.__x_test.size else X
            self.__y_test = np.concatenate([self.__y_test, y], axis = 0) if self.__y_test.size else y
            f.close()
        #self.__x_train, self.__x_test, self.__y_train, self.__y_test = x_train, x_test, y_train, y_test
        self.__model = self.build()
    
    #  model
    def build(self):
        inputImage = Input(shape=(image_shape))
        x = Conv2D(self.CNN_filters, kernel_size=(self.CNN_filter_size,self.CNN_filter_size), 
                   data_format="channels_last", strides=(1, 1), padding="same", input_shape=image_shape,
                    kernel_initializer='lecun_uniform', name='cnn2D_0')(inputImage)
        x = BatchNormalization()(x)
        x = Activation(self.activation[self.CNN_activation_index])(x)
        x = MaxPooling2D( pool_size = (self.CNN_MaxPool_size,self.CNN_MaxPool_size))(x)
        x = Dropout(self.dropout)(x)
        for i in range(1,self.CNN_layers):
            x = Conv2D(self.CNN_filters, kernel_size=(self.CNN_filter_size,self.CNN_filter_size), 
                   data_format="channels_last", strides=(1, 1), padding="same", input_shape=image_shape,
                    kernel_initializer='lecun_uniform', name='cnn2D_%i' %i)(x)
            x = BatchNormalization()(x)
            x = Activation(self.activation[self.CNN_activation_index])(x)
            #x = MaxPooling2D( pool_size = (self.CNN_MaxPool_size,self.CNN_MaxPool_size))(x)
            x = Dropout(self.dropout)(x)
            
        ####
        x = Flatten()(x)
        #
        for i in range(self.DNN_layers):
            x = Dense(self.DNN_neurons, activation=self.activation[self.DNN_activation_index], 
                      kernel_initializer='lecun_uniform', name='dense_%i' %i)(x)
            x = Dropout(self.dropout)(x)
        #
        output = Dense(5, activation='softmax', kernel_initializer='lecun_uniform', 
                       name = 'output_softmax')(x)
        ####
        model = Model(inputs=inputImage, outputs=output)
        model.compile(optimizer=self.optimizer[self.optimizer_index], 
                      loss='categorical_crossentropy', metrics=['acc'])
        return model

    
    # fit model
    def model_fit(self):
        #myTestGen = generator(self.input_test_files, self.batch_size)
        #myValGen = generator(self.input_val_files, self.batch_size)
        my_steps_per_epoch = int(10000/self.batch_size)
        myTestGen = DataGenerator(self.input_test_files, self.batch_size)
        myValGen = DataGenerator(self.input_val_files, self.batch_size)

        self.__model.fit_generator(generator=myTestGen, epochs=self.epochs*len(self.input_test_files), 
                                   steps_per_epoch= my_steps_per_epoch,  validation_data = myValGen,
                                   validation_steps = my_steps_per_epoch, verbose=0, 
                                   callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                                                           ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0), 
                                                           TerminateOnNaN()])
                                   
        #self.__model.fit(self.__x_train, self.__y_train,
        #                batch_size=self.batch_size,
        #                epochs=self.epochs,
        #                verbose=0,
        #                validation_data=[self.__x_test, self.__y_test],
        #                callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
        #                            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0),
        #                            TerminateOnNaN()])       
    
    # evaluate  model
    def model_evaluate(self):
        self.model_fit()
        evaluation = self.__model.evaluate(self.__x_test, self.__y_test, batch_size=self.batch_size, verbose=0)
        return evaluation


####################################################

# Runner function for model
# function to run  class

def run_model(inputTESTfiles, inputVALfiles, optmizer_index=0, CNN_filters=10, 
              CNN_filter_size=5, CNN_MaxPool_size=2, CNN_layers=2, CNN_activation_index=0, DNN_neurons=40, 
              DNN_layers=2, DNN_activation_index=0, dropout=0.2, batch_size=100, epochs=50):
    
    _model = myModel(inputTESTfiles, inputVALfiles, optmizer_index, CNN_filters, CNN_MaxPool_size, CNN_filter_size,
                 CNN_layers, CNN_activation_index, DNN_neurons, DNN_layers, DNN_activation_index, 
                 dropout, batch_size, epochs)
    model_evaluation = _model.model_evaluate()
    return model_evaluation

####################################################

#import glob
#X_test = np.array([])
#Y_test = np.array([])
#for fileIN in glob.glob("/eos/project/d/dshep/hls-fml/NEWDATA/VALIDATION/jetImage*_%sp*.h5" %sys.argv[1]):
#    print(fileIN)
#    f = h5py.File(fileIN, 'r')
#    myFeatures = np.array(f.get('jetImage'))
#    myTarget = np.array(f.get('jets')[0:,-6:-1])
#    X_test = np.concatenate([X_test,myFeatures], axis = 0) if X_test.size else myFeatures
#    Y_test = np.concatenate([Y_test,myTarget], axis = 0) if Y_test.size else myTarget
#    print(X_test.shape, Y_test.shape)
#print(X_test.shape, Y_test.shape)
#X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

inputTestFiles = glob.glob("/eos/project/d/dshep/hls-fml/NEWDATA/jetImage*_%sp*.h5" %sys.argv[1])
inputValFiles = glob.glob("/eos/project/d/dshep/hls-fml/NEWDATA/VALIDATION/jetImage*_%sp*.h5" %sys.argv[1])
n_epochs = 5000
img_rows = 100#X_train.shape[1]
img_cols = 100#X_train.shape[2]
image_shape = (img_rows, img_cols, 1)

# Bayesian Optimization

# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'optmizer_index',        'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'CNN_filters',           'type': 'discrete',   'domain': (10, 15, 20, 25, 30)},
          {'name': 'CNN_filter_size',       'type': 'discrete',   'domain': (3, 5, 7, 9)},
          {'name': 'CNN_MaxPool_size',      'type': 'discrete',   'domain': (2, 3, 5)},
          {'name': 'CNN_layers',            'type': 'discrete',   'domain': (1, 2, 3)},
          {'name': 'CNN_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'DNN_neurons',           'type': 'discrete',   'domain': (10, 20, 30, 40, 50, 60)},
          {'name': 'DNN_layers',            'type': 'discrete',   'domain': (1, 2, 3)},
          {'name': 'DNN_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'dropout',               'type': 'continuous', 'domain': (0.1, 0.4)},
          {'name': 'batch_size',            'type': 'discrete',   'domain': (50, 100, 200, 500)}]

# function to optimize model
def f(x):
    print(x)
    evaluation = run_model(inputTestFiles, inputValFiles,
                           optmizer_index = int(x[:,0]), 
                           CNN_filters = int(x[:,1]), 
                           CNN_filter_size = int(x[:,2]),
                           CNN_MaxPool_size = int(x[:,3]),
                           CNN_layers = int(x[:,4]), 
                           CNN_activation_index = int(x[:,5]), 
                           DNN_neurons = int(x[:,6]), 
                           DNN_layers = int(x[:,7]),
                           DNN_activation_index = int(x[:,8]),
                           dropout = float(x[:,9]),
                           batch_size = int(x[:,10]),
                           epochs = n_epochs)
    print("LOSS:\t{0} \t ACCURACY:\t{1}".format(evaluation[0], evaluation[1]))
    print(evaluation)
    return evaluation[0]

# run optimization
opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
opt_model.run_optimization(max_iter=10000)

print("DONE")
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
\t{14}:\t{15}
\t{16}:\t{17}
\t{18}:\t{19}
\t{20}:\t{21}
""".format(bounds[0]["name"],opt_model.x_opt[0],
           bounds[1]["name"],opt_model.x_opt[1],
           bounds[2]["name"],opt_model.x_opt[2],
           bounds[3]["name"],opt_model.x_opt[3],
           bounds[4]["name"],opt_model.x_opt[4],
           bounds[5]["name"],opt_model.x_opt[5],
           bounds[6]["name"],opt_model.x_opt[6],
           bounds[7]["name"],opt_model.x_opt[7],
           bounds[8]["name"],opt_model.x_opt[8],
           bounds[9]["name"],opt_model.x_opt[9],
           bounds[10]["name"],opt_model.x_opt[10]))
print("optimized loss: {0}".format(opt_model.fx_opt))

