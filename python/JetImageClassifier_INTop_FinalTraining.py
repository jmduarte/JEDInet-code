import setGPU
import os
import numpy as np
import h5py
import glob
import itertools
import sys
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim

from generatorINTop import InEventLoaderTop
import random

import tqdm

from gnn_top import GraphNetOld as GraphNet                                                                                    
#from gnn_top import GraphNet

args_cuda = bool(sys.argv[2])
args_sumO = bool(int(sys.argv[3])) if len(sys.argv)>3 else False


loc='IN_Top_Hyper_Old_%s'%(sys.argv[1])
import os
os.system('mkdir -p %s'%loc)
    
def get_sample(training, target, choice):
    target_vals = np.argmax(target, axis = 1)
    ind, = np.where(target_vals == choice)
    chosen_ind = np.random.choice(ind, 50000)
    return training[chosen_ind], target[chosen_ind]

def accuracy(predict, target):
    _, p_vals = torch.max(predict, 1)
    r = torch.sum(target == p_vals.squeeze(0)).cpu().data.numpy()
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

best_perf = {
    # hidden, De, Do, fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0
    150 : [64., 64., 16., 0., 2., 0., 0.]
}
sumO_best_perf = {
    # hidden, De, Do, fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0
    150 : [256., 64.,  32.,  2.,  0.,  2.,  0.] #optimized loss: 0.17599504769292157
}
# ### Prepare Dataset
nParticles = int(sys.argv[1])
x = sumO_best_perf[nParticles] if args_sumO else best_perf[nParticles]

labels = ['isTop','isQCD']
params = ['part_px', 'part_py' , 'part_pz' , 
          'part_energy' , 'part_erel' , 'part_pt' , 'part_ptrel', 
          'part_eta' , 'part_etarel' ,
          'part_eta_rot' , 
          'part_phi' , 'part_phirel' , 
          'part_phi_rot', 'part_deltaR',
          'part_costheta' , 'part_costhetarel']

batch_size = 64
n_epochs = 1000
patience = 10

import glob
import os

if os.path.isdir('/bigdata/shared'):
    inputTrainFiles = glob.glob("/bigdata/shared/JetImages/converted/rotation_224_150p_v1/train_file_*.h5")
    inputValFiles = glob.glob("/bigdata/shared/JetImages/converted/rotation_224_150p_v1/val_file_*.h5")

# hidden, De, Do, fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0
mymodel = GraphNet(nParticles, len(labels), params, int(x[0]), int(x[1]), int(x[2]), 
                   fr_activation=int(x[3]),  fo_activation=int(x[4]), fc_activation=int(x[5]), optimizer=int(x[6]), 
                   verbose=True, sum_O=args_sumO)

loss = nn.CrossEntropyLoss(reduction='mean')
if mymodel.optimizer == 1:        
    optimizer = optim.Adadelta(mymodel.parameters(), lr = 0.0001)
else:
    optimizer = optim.Adam(mymodel.parameters(), lr = 0.0001)
loss_train = np.zeros(n_epochs)
loss_val = np.zeros(n_epochs)

#random.shuffle(inputTrainFiles)
#random.shuffle(inputValFiles)
# Define the data generators from the training set and validation set.
train_set = InEventLoaderTop(file_names=inputTrainFiles, nP=nParticles,
                             feature_names = params,label_name = 'label', verbose=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_set = InEventLoaderTop(file_names=inputValFiles, nP=nParticles,
                        feature_names = params,label_name = 'label', verbose=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

nBatches_per_training_epoch = len(train_set)/batch_size
nBatches_per_validation_epoch = len(val_set)/batch_size
print("nBatches_per_training_epoch: %i" %nBatches_per_training_epoch)
print("nBatches_per_validation_epoch: %i" %nBatches_per_validation_epoch)

best_loss_val = 9999
stale_epochs = 0
for i in range(n_epochs):
    if mymodel.verbose: print("Epoch %s" % i)
    # train
    t = tqdm.tqdm(enumerate(train_loader),total=len(train_set)/batch_size)
    mymodel.train()   
    for batch_idx, mydict in t:
        data = mydict['jetConstituentList']
        target = mydict['jets']
        if args_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        out = mymodel(data)
        l = loss(out, target)
        acc = accuracy(out, target)
        l.backward()
        optimizer.step()
        loss_train_item = l.cpu().data.numpy()
        loss_train[i] += loss_train_item/nBatches_per_training_epoch
        t.set_description("train batch loss = %.5f, acc = %.5f" % (loss_train_item, acc))
        t.refresh() # to show immediately the update
    # validation
    v = tqdm.tqdm(enumerate(val_loader), total = len(val_set)/batch_size)
    mymodel.eval()
    with torch.no_grad():
        out_vals = []
        targets = []
        for batch_idx, mydict in v:
            data = mydict['jetConstituentList']
            target = mydict['jets']
            if args_cuda:
                data, target = data.cuda(), target.cuda()
            out_val = mymodel(data)
            out_vals  += [out_val]
            targets  += [target]
            l_val = loss(out_val, target)
            acc_val = accuracy(out_val, target)
            loss_val_item = l_val.cpu().data.numpy()
            loss_val[i] += loss_val_item/nBatches_per_validation_epoch
            v.set_description("val batch loss = %.5f, acc = %.5f" % (loss_val_item, acc_val))
            v.refresh() # to show immediately the update
        targets = torch.cat(targets,0)
        out_vals = torch.cat(out_vals,0)
        acc_vals = accuracy(out_vals, targets)
    if mymodel.verbose: 
        print("Training   Loss: %f" %loss_train[i])
    if mymodel.verbose: 
        print("Validation Loss: %f" %loss_val[i])
        print("Validation Acc: %f" %acc_vals)
    if loss_val[i] < best_loss_val:
        best_loss_val = loss_val[i]
        print("Best new model")
        # save training
        torch.save(mymodel.state_dict(), "%s/IN%s_bestmodel.params" %(loc, '_sumO' if mymodel.sum_O else ''))
    else:
        print("Stale epoch")
        stale_epochs += 1
        if stale_epochs>=patience:
            print("Early Stopping at",i)
            # the last model
            torch.save(mymodel.state_dict(), "%s/IN%s_lastmodel.params" %(loc, '_sumO' if mymodel.sum_O else ''))
            break

# save training history
f = h5py.File("%s/history%s.h5" %(loc, '_sumO' if mymodel.sum_O else ''), "w")
f.create_dataset('train_loss', data= np.asarray(loss_train), compression='gzip')
f.create_dataset('val_loss', data= np.asarray(loss_val), compression='gzip')

