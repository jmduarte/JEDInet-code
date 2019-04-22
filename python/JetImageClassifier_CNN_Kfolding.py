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

# For ROC curves
import pandas as pd
from sklearn.metrics import roc_curve, auc

####################################################

CNN_filters = 10
CNN_filter_size = 3
CNN_MaxPool_size = 5
CNN_layers = 1
CNN_activation = 'elu'
DNN_neurons = 50
DNN_layers = 3
DNN_activation = 'elu'
dropout = 0.1
batch_size = 500
n_epochs = 500
labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
nParticles = 50

#  model
def myModel():
    inputImage = Input(shape=(image_shape))
    x = Conv2D(CNN_filters, kernel_size=(CNN_filter_size,CNN_filter_size), 
               data_format="channels_last", strides=(1, 1), padding="same", input_shape=image_shape,
               kernel_initializer='lecun_uniform', name='cnn2D_0')(inputImage)
    x = BatchNormalization()(x)
    x = Activation(CNN_activation)(x)
    x = MaxPooling2D( pool_size = (CNN_MaxPool_size,CNN_MaxPool_size))(x)
    x = Dropout(dropout)(x)
    for i in range(1,CNN_layers):
        x = Conv2D(CNN_filters, kernel_size=(CNN_filter_size,CNN_filter_size), 
                   data_format="channels_last", strides=(1, 1), padding="same", input_shape=image_shape,
                   kernel_initializer='lecun_uniform', name='cnn2D_%i' %i)(x)
        x = BatchNormalization()(x)
        x = Activation(CNN_activation)(x)
        #x = MaxPooling2D( pool_size = (CNN_MaxPool_size,CNN_MaxPool_size))(x)
        x = Dropout(dropout)(x)
        
    ####
    x = Flatten()(x)
    #
    for i in range(DNN_layers):
        x = Dense(DNN_neurons, activation=DNN_activation, 
                  kernel_initializer='lecun_uniform', name='dense_%i' %i)(x)
        x = Dropout(dropout)(x)
    #
    output = Dense(5, activation='softmax', kernel_initializer='lecun_uniform', 
                   name = 'output_softmax')(x)
    ####
    model = Model(inputs=inputImage, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model

inputFiles = glob.glob("/data/ml/mpierini/hls-fml/jetImage*_%sp*.h5" %nParticles)
#inputFiles = glob.glob("/data/ml/mpierini/hls-fml/jetImage_1_%sp_[1-4]*" %nParticles)

#######
img_rows = 100#X.shape[1]
img_cols = 100#X.shape[2]
image_shape = (img_rows, img_cols, 1)

models = []
histories = []
from sklearn.model_selection import train_test_split
# k-folding
k_index = 10
nFilesVal = int(len(inputFiles)/(k_index+1))
for k in range(k_index):
    testFileFirst = k*nFilesVal
    testFileLast = min((k+1)*nFilesVal, len(inputFiles))
    inputValFiles = []
    inputTrainFiles = inputFiles.copy()
    for iFile in range(testFileFirst,testFileLast):
        inputValFiles.append(inputFiles[iFile])
        inputTrainFiles.remove(inputFiles[iFile])
    # prepare generator
    my_batch_per_file = int(10000/batch_size)
    myTestGen = DataGenerator("TRAINING", inputTrainFiles, batch_size, my_batch_per_file)
    myValGen = DataGenerator("VALIDATION", inputValFiles, batch_size, my_batch_per_file)
    # now fit
    thisModel = myModel()
    print("Nepochs = %i" %n_epochs)
    print("Steps per training epoch = %i" %(my_batch_per_file*len(inputTrainFiles)))
    print("Steps per validation epoch = %i" %(my_batch_per_file*len(inputValFiles)))
    thisHistory = thisModel.fit_generator(generator=myTestGen, epochs=n_epochs,
                                          steps_per_epoch= my_batch_per_file*len(inputTrainFiles),  validation_data = myValGen,
                                          validation_steps = my_batch_per_file*len(inputValFiles), verbose=0,
                                          callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                                                       ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0),
                                                       TerminateOnNaN()])
    models.append(thisModel)
    histories.append(thisHistory)

# create the test dataset  
inputFiles = glob.glob("/data/ml/mpierini/hls-fml/VALIDATION/jetImage_9_%sp*.h5" %nParticles)
X_test = np.array([])
Y_test = np.array([])
for fileINname in inputFiles:
    fileIN =  h5py.File(fileINname, "r")
    thisX = np.array(fileIN.get('jetImage'))
    thisY = np.array(fileIN.get('jets')[0:,-6:-1])
    Y_test = np.concatenate([Y_test, thisY], axis = 0) if Y_test.size else thisY
    X_test = np.concatenate([X_test, thisX], axis = 0) if X_test.size else thisX
X_test=  X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

#### get the ROC curves
predict_test = []
for k in range(len(inputFiles)):
    predict_test.append(models[k].predict(X_test))

df = pd.DataFrame()
fpr = {}
tpr = {}
auc1 = {}
for i, label in enumerate(labels):

        df[label] = Y_test[:,i]
        for k in range(len(inputFiles)):
            df[label + '_pred_%i' %k] = predict_test[k][:,i]
            print("%s_%i" %(label,k))
            fpr["%s_%i" %(label,k)], tpr["%s_%i" %(label,k)], threshold = roc_curve(df[label],df[label+'_pred_%i' %k])
            auc1["%s_%i" %(label,k)] = auc(fpr["%s_%i" %(label,k)], tpr["%s_%i" %(label,k)])

# SAVE DATA FRAMES in a new file
import pickle

with open('%s/CNN_ROC_fpr.pickle' %sys.argv[1], 'wb') as handle:
    pickle.dump(fpr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('%s/CNN_ROC_tpr.pickle' %sys.argv[1], 'wb') as handle:
    pickle.dump(tpr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('%s/CNN_ROC_AUC.pickle' %sys.argv[1], 'wb') as handle:
    pickle.dump(auc1, handle, protocol=pickle.HIGHEST_PROTOCOL)





