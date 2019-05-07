import setGPU
import os
import numpy as np
import h5py
import glob
import itertools
import sys
from sklearn.utils import shuffle
import random
from tqdm import tqdm

# hyperparameters
import GPy, GPyOpt

import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim
from generatorIN import InEventLoader

args_sumO = bool(sys.argv[3]) if len(sys.argv)>3 else False

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

        self.sum_O = args_sumO
        self.Ra = torch.ones(self.Dr, self.Nr)
        self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden).cuda()
        self.fr2 = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fr3 = nn.Linear(int(hidden/2), self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden).cuda()
        self.fo2 = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fo3 = nn.Linear(int(hidden/2), self.Do).cuda()
        if self.sum_O:
            self.fc1 = nn.Linear(self.Do *1, hidden).cuda()
        else:
            self.fc1 = nn.Linear(self.Do * self.N, hidden).cuda()
        self.fc2 = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fc3 = nn.Linear(int(hidden/2), self.n_targets).cuda()

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = Variable(self.Rr).cuda()
        self.Rs = Variable(self.Rs).cuda()

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
        if self.sum_O:
            O = torch.sum( O, dim=1)
        ### Classification MLP ###
        if self.fc_activation ==2:
            if self.sum_O:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.selu(self.fc1(O.view(-1, self.Do * self.N)))

            N = nn.functional.selu(self.fc2(N))       
        elif self.fc_activation ==1:
            if self.sum_O:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * 1)))
            else:
                N = nn.functional.elu(self.fc1(O.view(-1, self.Do * self.N)))
            N = nn.functional.elu(self.fc2(N))
        else:
            if self.sum_O:
                N = nn.functional.relu(self.fc1(O.view(-1, self.Do * 1)))
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

####################
    
def get_sample(training, target, choice):
    target_vals = np.argmax(target, axis = 1)
    ind, = np.where(target_vals == choice)
    chosen_ind = np.random.choice(ind, 50000)
    return training[chosen_ind], target[chosen_ind]

def accuracy(predict, target):
    _, p_vals = torch.max(predict, 1)
    r = torch.sum(target == p_vals.squeeze(1)).data.numpy()[0]
    t = target.size()[0]
    return r * 1.0 / t

def stats(predict, target):
    print(predict)
    _, p_vals = torch.max(predict, 1)
    t = target.cpu().data.numpy()
    p_vals = p_vals.squeeze(1).data.numpy()
    vals = np.unique(t)
    for i in vals:
        ind = np.where(t == i)
        pv = p_vals[ind]
        correct = sum(pv == t[ind])
        print("  Target %s: %s/%s = %s%%" % (i, correct, len(pv), correct * 100.0/len(pv)))
    print("Overall: %s/%s = %s%%" % (sum(p_vals == t), len(t), sum(p_vals == t) * 100.0/len(t)))
    return sum(p_vals == t) * 100.0/len(t)


# ### Prepare Dataset

nParticles = int(sys.argv[1])
args_cuda = bool(sys.argv[2])
labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
params = ['j1_px', 'j1_py' , 'j1_pz' , 'j1_e' , 'j1_erel' , 'j1_pt' , 'j1_ptrel', 'j1_eta' , 'j1_etarel' , 
          'j1_etarot' , 'j1_phi' , 'j1_phirel' , 'j1_phirot', 'j1_deltaR' , 'j1_costheta' , 'j1_costhetarel']

# Load
#X = np.array([])
#Y = np.array([])
#First = True
#for fileIN in glob.glob("/data/ML/mpierini/hls-fmljetImage*_%sp*.h5" %sys.argv[1]):
#    print(fileIN)
#    f = h5py.File(fileIN, 'r')
#    myFeatures = np.array(f.get('jetConstituentList'))
#    myTarget = np.array(f.get('jets')[0:,-6:-1])
#    print(myFeatures.size)
#    X = np.concatenate([X,myFeatures], axis = 0) if X.size else myFeatures
#    Y = np.concatenate([Y,myTarget], axis = 0) if Y.size else myTarget
#    print(X.shape, Y.shape)

val_split = 0.3
batch_size = 100
n_epochs = 10000
patience = 10

# cut dataset so that # examples int(examples / batch size)
#new_Nexamples = int(X.shape[0]/batch_size)*batch_size
#X = X[:new_Nexamples, :,:]
#Y = Y[:new_Nexamples]
#print(X.shape, Y.shape)

# transpose constituents index and feature index (to match what IN expects)
#X = np.swapaxes(X, 1, 2)
# pytorch Cross Entropy doesn't support one-hot encoding
#Y = np.argmax(Y, axis=1)

# shuffle and split
#X, Y = shuffle(X, Y, random_state=0)

#Nval = int(X.shape[0]*val_split)
#X_val = X[:Nval,:,:]
#X_train = X[Nval:,:,:]
#Y_val = Y[:Nval]
#Y_train = Y[Nval:]
# epochs
#n_batches_train = int(X_train.shape[0]/batch_size)
#n_batches_val = int(X_val.shape[0]/batch_size)
#print(X_train.shape, Y_train.shape)
#print(X_val.shape, Y_val.shape)

# Convert dataset to pytorch
#X_train = Variable(torch.FloatTensor(X_train))
#X_val = Variable(torch.FloatTensor(X_val))
#Y_train = Variable(torch.LongTensor(Y_train).long())  
#Y_val = Variable(torch.LongTensor(Y_val).long())  

# Bayesian Optimization

# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'hidden_neurons',       'type': 'discrete',   'domain': (6, 10, 20, 30, 40, 50)},
          {'name' : 'De',                  'type': 'discrete',   'domain': (4, 6, 8, 10, 12, 14)},
          {'name' : 'Do',                  'type': 'discrete',   'domain': (4, 6, 8, 10, 12, 14)},
          {'name': 'fr_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'fo_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'fc_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'optmizer_index',       'type': 'discrete',   'domain': (0, 1)}]

# model-training function

def model_evaluate(mymodel):
    #loss = nn.CrossEntropyLoss(reduction='sum')
    loss = nn.CrossEntropyLoss(reduction='mean')
    if mymodel.optimizer == 1:        
        optimizer = optim.Adadelta(mymodel.parameters(), lr = 0.0001)
    else:
        optimizer = optim.Adam(mymodel.parameters(), lr = 0.0001)
    loss_train = np.zeros(n_epochs)
    loss_val = np.zeros(n_epochs)

    # Define the data generators from the training set and validation set. Let's try a batch size of 12.
    import glob
    #inputTrainFiles = glob.glob("/data/ML/mpierini/hls-fml/jetImage*_%sp*.h5" %nParticles)
    #inputValFiles = glob.glob("/data/ML/mpierini/hls-fml/VALIDATION/jetImage*_%sp*.h5" %nParticles)
    import os
    if os.path.isdir('/imdata'):
        inputTrainFiles = glob.glob("/imdata/NEWDATA/jetImage*_%sp*.h5" %nParticles)
        inputValFiles = glob.glob("/imdata/NEWDATA/VALIDATION/jetImage*_%sp*.h5" %nParticles)
    elif os.path.isdir('/home/jduarte'):
        inputTrainFiles = glob.glob("/home/jduarte/NEWDATA/jetImage*_%sp*.h5" %nParticles)
        inputValFiles = glob.glob("/home/jduarte/NEWDATA/VALIDATION/jetImage*_%sp*.h5" %nParticles)
    elif os.path.isdir('/bigdata/shared'):
        inputTrainFiles = glob.glob("/bigdata/shared/hls-fml/NEWDATA/jetImage*_%sp*.h5" %nParticles)
        inputValFiles = glob.glob("/bigdata/shared/hls-fml/NEWDATA/VALIDATION/jetImage*_%sp*.h5" %nParticles)        

    random.shuffle(inputTrainFiles)
    random.shuffle(inputValFiles)

    nBatches_per_training_epoch = len(inputTrainFiles)*10000/batch_size
    nBatches_per_validation_epoch = len(inputValFiles)*10000/batch_size

    train_set = InEventLoader(file_names=inputTrainFiles, nP=nParticles,
                              feature_name ='jetConstituentList',label_name = 'jets', verbose=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_set = InEventLoader(file_names=inputValFiles, nP=nParticles,
                            feature_name ='jetConstituentList',label_name = 'jets', verbose=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    for i in range(n_epochs):
        if mymodel.verbose: print("Epoch %s" % i)
        #for j in range(0, xtrain.size()[0], batch_size):
        #for (batch_idx, mydict) in tqdm(enumerate(train_loader),total=nBatches_per_training_epoch):
        for (batch_idx, mydict) in enumerate(train_loader):
            data = mydict['jetConstituentList']
            target = mydict['jets']
            if args_cuda:
                data, target = data.cuda(), target.cuda()
            #### ?????
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            out = mymodel(data)
            l = loss(out, target)
            l.backward()
            optimizer.step()
            loss_train[i] += l.cpu().data.numpy()/nBatches_per_training_epoch
        #loss_train[i] = loss_train[i]/float(xtrain.size()[0])
        #for j in range(0, xval.size()[0], batch_size):
        #for (batch_idx, mydict) in tqdm(enumerate(val_loader),total=nBatches_per_validation_epoch):
        for (batch_idx, mydict) in enumerate(val_loader):
            data = mydict['jetConstituentList']
            target = mydict['jets']
            if args_cuda:
                data, target = data.cuda(), target.cuda()
            #### ?????
            data, target = Variable(data, volatile=True), Variable(target)
            out_val = mymodel(data)
            l_val = loss(out_val, target)
            loss_val[i] += l_val.cpu().data.numpy()/nBatches_per_validation_epoch
        #loss_val[i] = loss_val[i]/float(xval.size()[0])
        if mymodel.verbose: print("Training   Loss: %f" %loss_train[i])
        if mymodel.verbose: print("Validation Loss: %f" %loss_val[i])
        #that below does not trigger soon enough
        if all(loss_val[max(0, i - patience):i] > min(np.append(loss_val[0:max(0, i - patience)], 200))) and i > patience:
            print("Early Stopping")
            break
        #that below does not trigger soon enough        
        if i > (2*patience):
            last_avg = np.mean(loss_val[i - patience:i])
            previous_avg = np.mean(loss_val[i - 2*patience : i - patience])
            if last_avg > previous_avg:
                print("Early Avg Stopping")
                break
        if i > patience:
            last_min = min(loss_val[i - patience:i])
            overall_min = min(loss_val)
            if last_min > overall_min:
                print("Early min Stopping")
                break
    loss_val = loss_val[loss_val>0]
    return loss_val[-1]

# function to optimize model
def f(x):
    print(x)
    gnn = GraphNet(nParticles, len(labels), params, int(x[:,0]), int(x[:,1]), int(x[:,2]), 
                   int(x[:,3]),  int(x[:,4]),  int(x[:,5]), int(x[:,6]))
    val_loss = model_evaluate(gnn)
    print("LOSS: %f" %val_loss)
    return val_loss


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

