#!/usr/bin/env python

import sys

withGPU = False
if len(sys.argv)>1:
    if sys.argv[1] == "--withGPU":  withGPU = True
if withGPU: import setGPU

# In[1]:


import numpy as np
import h5py
import time 


# In[2]:


import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
import itertools


# In[3]:


labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']



# input dataset
import glob
X_hlf = np.array([])
Y = np.array([])
file_150 = h5py.File("../data/jetImage_7_150p_10000_20000.h5","r")

HLF = np.array(file_150.get('jets')[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]])
Image = np.array(file_150.get('jetImage'))
Image = Image.reshape((Image.shape[0], Image.shape[1], Image.shape[2], 1))
List_150 =  np.array(file_150.get('jetConstituentList'))
List_50  = List_150[:,:50,:]
List_100  = List_150[:,:100,:]


# In[6]:


print(HLF.shape, Image.shape, List_50.shape, List_100.shape, List_150.shape)


# In[7]:


List_150 = np.swapaxes(List_150, 1, 2)
List_150tf = torch.FloatTensor(List_150)
List_100 = np.swapaxes(List_100, 1, 2)
List_100tf = torch.FloatTensor(List_100)


# # DNN

# In[23]:


# keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, Dropout, Flatten
from keras.layers import Concatenate, BatchNormalization, Activation
from keras.layers import MaxPooling2D, MaxPooling3D, GRU
from keras.utils import plot_model
from keras import regularizers
from keras import backend as K
from keras import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.regularizers import l1


# In[24]:


#optmizer = 'adam'
# best model from optimization
DNN_neurons = 80
DNN_layers = 3
DNN_activation = 'elu'
dropout = 0.10
batch_size = 50
n_epochs = 500


# In[25]:


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

    output = Dense(5, activation='softmax', kernel_initializer='lecun_uniform', 
                   name = 'output_softmax')(x)
    ####
    model = Model(inputs=inputArray, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


# In[26]:


input_shape = len([12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52])
thisModel = myModel()
#thisModel.summary()


# In[27]:


time_before = float(time.time())
thisModel.predict(HLF)
time_after = float(time.time())
print(time_before)
average_time= (time_after-time_before)/HLF.shape[1]
print("DNN Average time = %.4f msec" %(1000.*average_time))


# # CNN

# In[28]:


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


# In[29]:


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


# In[30]:


img_rows = 100#X.shape[1]
img_cols = 100#X.shape[2]
image_shape = (img_rows, img_cols, 1)


# In[31]:


thisModel = myModel()
#thisModel.summary()


# In[32]:


time_before = float(time.time())
thisModel.predict(Image)
time_after = float(time.time())
print(time_before)
average_time= (time_after-time_before)/HLF.shape[1]
print("CNN Average time = %.4f msec" %(1000.*average_time))


# # GRU

# In[34]:


nParticles = 50
GRU_units= 50
DNN_neurons = 40
DNN_layers = 3
DNN_activation = 'relu'
dropout = 0.22
batch_size = 500
n_epochs = 50
#n_epochs = 1
labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
input_shape = (nParticles,16)


# In[35]:


#  model
def myModel():
    inputArray = Input(shape=(input_shape))
    x = GRU(GRU_units, activation='tanh',
            recurrent_activation='hard_sigmoid', name='gru')(inputArray)
    x = Dropout(dropout)(x)
    ####
    for i in range(0,DNN_layers):
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


# In[36]:


thisModel = myModel()
#thisModel.summary()


# In[37]:


time_before = float(time.time())
thisModel.predict(List_50)
time_after = float(time.time())
print(time_before)
average_time= (time_after-time_before)/HLF.shape[1]
print("GRU Average time = %.4f msec" %(1000.*average_time))


# # JEDI-net

# In[38]:


class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De, Do, 
                 fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0, verbose = False):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation
        self.optimizer = optimizer
        self.verbose = verbose
        self.assign_matrices()

        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden)
        self.fr2 = nn.Linear(hidden, int(hidden/2))
        self.fr3 = nn.Linear(int(hidden/2), self.De)
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden)
        self.fo2 = nn.Linear(hidden, int(hidden/2))
        self.fo3 = nn.Linear(int(hidden/2), self.Do)
        self.fc1 = nn.Linear(self.Do * self.N, hidden)
        self.fc2 = nn.Linear(hidden, int(hidden/2))
        self.fc3 = nn.Linear(int(hidden/2), self.n_targets)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = Variable(self.Rr)
        self.Rs = Variable(self.Rs)

    def forward(self, x):
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        if self.fr_activation ==2:
            B = nn.functional.selu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.selu(self.fr2(B))
            E = nn.functional.selu(self.fr3(B).view(-1, self.Nr, self.De))            
        elif self.fr_activation ==1:
            B = nn.functional.elu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.elu(self.fr2(B))
            E = nn.functional.elu(self.fr3(B).view(-1, self.Nr, self.De))
        else:
            B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.relu(self.fr2(B))
            E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x, Ebar], 1)
        del Ebar
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        if self.fo_activation ==2:
            C = nn.functional.selu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.selu(self.fo2(C))
            O = nn.functional.selu(self.fo3(C).view(-1, self.N, self.Do))
        elif self.fo_activation ==1:
            C = nn.functional.elu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.elu(self.fo2(C))
            O = nn.functional.elu(self.fo3(C).view(-1, self.N, self.Do))
        else:
            C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.relu(self.fo2(C))
            O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ### Classification MLP ###
        if self.fc_activation ==2:
            N = nn.functional.selu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.selu(self.fc2(N))       
        elif self.fc_activation ==1:
            N = nn.functional.elu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.elu(self.fc2(N))
        else:
            N = nn.functional.relu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.relu(self.fc2(N))
        del O
        #N = nn.functional.relu(self.fc3(N))
        N = self.fc3(N)
        return N

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


# In[39]:


# ### Prepare Dataset                                                                                                                                                  
nParticles = 100
x = []
x.append(30) # hinned nodes                                                                                                                                            
x.append(10) # De                                                                                                                                                      
x.append(10) # Do                                                                                                                                                      
x.append(1) # fr_activation_index                                                                                                                                      
x.append(1) # fo_activation_index                                                                                                                                      
x.append(1) # fc_activation_index                                                                                                                                      
x.append(0) # optmizer_index           


# In[40]:


params = ['j1_px', 'j1_py' , 'j1_pz' , 'j1_e' , 'j1_erel' , 'j1_pt' , 'j1_ptrel', 'j1_eta' , 'j1_etarel' , 
          'j1_etarot' , 'j1_phi' , 'j1_phirel' , 'j1_phirot', 'j1_deltaR' , 'j1_costheta' , 'j1_costhetarel']


# In[41]:


mymodel = GraphNet(nParticles, len(labels), params, int(x[0]), int(x[1]), int(x[2]), 
                       int(x[3]),  int(x[4]),  int(x[5]), int(x[6]), 0)


# In[42]:


#time_before = float(time.time())
#mymodel(List_150)
#time_after = float(time.time())
#print(time_before)
#average_time= (time_after-time_before)/HLF.shape[1]
#print("IN time = %.4f msec" %average_time*1000.)


# # JEDI-net with Sum over O

# In[43]:


class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, De, Do, 
                 fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0, verbose = False):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = len(params)
        self.N = n_constituents
        self.Nr = self.N * (self.N - 1)
        self.Dr = 0
        self.De = De
        self.Dx = 0
        self.Do = Do
        self.n_targets = n_targets
        self.fr_activation = fr_activation
        self.fo_activation = fo_activation
        self.fc_activation = fc_activation
        self.optimizer = optimizer
        self.verbose = verbose
        self.assign_matrices()

        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden)
        self.fr2 = nn.Linear(hidden, int(hidden/2))
        self.fr3 = nn.Linear(int(hidden/2), self.De)
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden)
        self.fo2 = nn.Linear(hidden, int(hidden/2))
        self.fo3 = nn.Linear(int(hidden/2), self.Do)
        self.fc1 = nn.Linear(self.Do, hidden)
        self.fc2 = nn.Linear(hidden, int(hidden/2))
        self.fc3 = nn.Linear(int(hidden/2), self.n_targets)

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = Variable(self.Rr)
        self.Rs = Variable(self.Rs)

    def forward(self, x):
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        if self.fr_activation ==2:
            B = nn.functional.selu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.selu(self.fr2(B))
            E = nn.functional.selu(self.fr3(B).view(-1, self.Nr, self.De))            
        elif self.fr_activation ==1:
            B = nn.functional.elu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.elu(self.fr2(B))
            E = nn.functional.elu(self.fr3(B).view(-1, self.Nr, self.De))
        else:
            B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
            B = nn.functional.relu(self.fr2(B))
            E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x, Ebar], 1)
        del Ebar
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        if self.fo_activation ==2:
            C = nn.functional.selu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.selu(self.fo2(C))
            O = nn.functional.selu(self.fo3(C).view(-1, self.N, self.Do))
        elif self.fo_activation ==1:
            C = nn.functional.elu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.elu(self.fo2(C))
            O = nn.functional.elu(self.fo3(C).view(-1, self.N, self.Do))
        else:
            C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + self.De)))
            C = nn.functional.relu(self.fo2(C))
            O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C
        ## sum over the O matrix  
        O = torch.sum( O, dim=1)
        ### Classification MLP ###
        if self.fc_activation ==2:
            N = nn.functional.selu(self.fc1(O.view(-1, self.Do)))
            N = nn.functional.selu(self.fc2(N))       
        elif self.fc_activation ==1:
            N = nn.functional.elu(self.fc1(O.view(-1, self.Do)))
            N = nn.functional.elu(self.fc2(N))
        else:
            N = nn.functional.relu(self.fc1(O.view(-1, self.Do)))
            N = nn.functional.relu(self.fc2(N))
        del O
        #N = nn.functional.relu(self.fc3(N))
        N = self.fc3(N)
        return N

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


# In[44]:


# ### Prepare Dataset                                                                                                                                                  
nParticles = 150
x = []
x.append(50) # hinned nodes                                                                                                                                            
x.append(14) # De                                                                                                                                                      
x.append(12) # Do                                                                                                                                                      
x.append(2) # fr_activation_index                                                                                                                                      
x.append(2) # fo_activation_index                                                                                                                                      
x.append(2) # fc_activation_index                                                                                                                                      
x.append(0) # optmizer_index           


# In[45]:

mymodel = GraphNet(nParticles, len(labels), params, int(x[0]), int(x[1]), int(x[2]), 
                       int(x[3]),  int(x[4]),  int(x[5]), int(x[6]), 0)

print("STARTING IN CONVERSION")


# convert to ONNX
torch.onnx.export(mymodel, List_150tf, "../models/IN_withSum.onnx")

print("DONE IN CONVERSION")


# convert from ONNX to Tensorflow

from onnx_tf.backend import prepare
model = onnx.load("../models/IN_withSum.onnx")
tf_rep = prepare(model)

print("MODEL IMPORTED IN TF")

time_before = float(time.time())
tf_rep.predict(List_50)
time_after = float(time.time())
print(time_before)
average_time= (time_after-time_before)/HLF.shape[1]
print("IN with Sum O Average time = %.4f msec" %average_time*1000.)



