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

# For ROC curves
import pandas as pd
from sklearn.metrics import roc_curve, auc


####################################################

#optmizer = 'adam'
# best model from optimization
DNN_neurons = 80
DNN_layers = 3
DNN_activation = 'elu'
dropout = 0.10
batch_size = 50
n_epochs = 500
labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']

#  model
def myModel():
    inputArray = Input(shape=(input_shape,))
    x = Dense(DNN_neurons, activation=DNN_activation, 
              kernel_initializer='lecun_uniform', name='dense_0')(inputArray)
    x = Dropout(dropout)(x)
    ####
    for i in range(1,DNN_layers):
        x = Dense(DNN_neurons, activation=DNN_activation, 
                  kernel_initializer='lecun_uniform', name='dense_%i' %i)(x)
        x = Dropout(dropout)(x)
    #
    output = Dense(5, activation='softmax', kernel_initializer='lecun_uniform', 
                   name = 'output_softmax')(x)
    ####
    model = Model(inputs=inputArray, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

import glob
X = np.array([])
Y = np.array([])
for fileIN in glob.glob("/data/ml/mpierini/hls-fml/jetImage*_30p*.h5"):
    print(fileIN)
    f = h5py.File(fileIN, 'r')
    myFeatures = np.array(f.get('jets')[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]])
    myTarget = np.array(f.get('jets')[0:,-6:-1])
    X = np.concatenate([X,myFeatures], axis = 0) if X.size else myFeatures
    Y = np.concatenate([Y,myTarget], axis = 0) if Y.size else myTarget
    print(X.shape, Y.shape)
input_shape = len([12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52])

####  NEED TO SHUFFLE X AND Y
from sklearn.utils import shuffle
X, Y = shuffle(X, Y, random_state=0)

#######
Nsamples = X.shape[0]

models = []
histories = []
from sklearn.model_selection import train_test_split
# k-folding
for k in range(10):
    print(k*int(Nsamples/10.),int(Nsamples/10.)*(k+1))
    X_test = X[k*int(Nsamples/10.):int(Nsamples/10.)*(k+1),:]
    Y_test = Y[k*int(Nsamples/10.):int(Nsamples/10.)*(k+1),:]
    X_train = np.concatenate([X[0:k*int(Nsamples/10.),:], X[int(Nsamples/10.)*(k+1):,:]], axis = 0)
    Y_train = np.concatenate([Y[0:k*int(Nsamples/10.),:], Y[int(Nsamples/10.)*(k+1):,:]], axis = 0)
    thisModel = myModel()
    thisHistory = thisModel.fit(X_train, Y_train, batch_size, epochs=n_epochs,
                                verbose=0, validation_data=[X_test, Y_test],
                                callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=0),
                                             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0),
                                             TerminateOnNaN()])
    models.append(thisModel)
    histories.append(thisHistory)

#### get the ROC curves
predict_test = []
for k in range(10):
    predict_test.append(models[k].predict(X_test))

df = pd.DataFrame()
fpr = {}
tpr = {}
auc1 = {}
for i, label in enumerate(labels):
        df[label] = Y_test[:,i]
        for k in range(10):
            df[label + '_pred_%i' %k] = predict_test[k][:,i]
            print("%s_%i" %(label,k))
            fpr["%s_%i" %(label,k)], tpr["%s_%i" %(label,k)], threshold = roc_curve(df[label],df[label+'_pred_%i' %k])
            auc1["%s_%i" %(label,k)] = auc(fpr["%s_%i" %(label,k)], tpr["%s_%i" %(label,k)])

# SAVE DATA FRAMES in a new file
import pickle

with open('%s/DNN_ROC_fpr.pickle' %sys.argv[1], 'wb') as handle:
    pickle.dump(fpr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('%s/DNN_ROC_tpr.pickle' %sys.argv[1], 'wb') as handle:
    pickle.dump(tpr, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('%s/DNN_ROC_AUC.pickle' %sys.argv[1], 'wb') as handle:
    pickle.dump(auc1, handle, protocol=pickle.HIGHEST_PROTOCOL)






