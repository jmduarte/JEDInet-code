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

# hyperparameters
import GPy, GPyOpt

from generatorINTop import InEventLoaderTop
import random

import tqdm

from gnn_top import GraphNetOld as GraphNet

args_cuda = bool(sys.argv[2])
args_sumO = bool(int(sys.argv[3])) if len(sys.argv)>3 else False


loc='IN_Top_Hyper_%s'%(sys.argv[1])
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

# ### Prepare Dataset
nParticles = int(sys.argv[1])

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
n_iter = 10
patience = 10

import glob
import os

# Bayesian Optimization

# the bounds dict should be in order of continuous type and then discrete type
bounds = [{'name': 'hidden_neurons',       'type': 'discrete',   'domain': (16, 32, 64, 128, 256)},
          {'name' : 'De',                  'type': 'discrete',   'domain': (4, 8, 16, 32, 64)},
          {'name' : 'Do',                  'type': 'discrete',   'domain': (4, 8, 16, 32, 64)},
          {'name': 'fr_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'fo_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'fc_activation_index',  'type': 'discrete',   'domain': (0, 1, 2)},
          {'name': 'optmizer_index',       'type': 'discrete',   'domain': (0, 1)}]

def model_evaluate(mymodel, param_string):
    loss = nn.CrossEntropyLoss(reduction='mean')
    if mymodel.optimizer == 1:        
        optimizer = optim.Adadelta(mymodel.parameters(), lr = 0.0001)
    else:
        optimizer = optim.Adam(mymodel.parameters(), lr = 0.0001)
    loss_train = np.zeros(n_epochs)
    loss_val = np.zeros(n_epochs)

    if os.path.isdir('/bigdata/shared'):
        inputTrainFiles = glob.glob("/bigdata/shared/JetImages/converted/rotation_224_150p_v1/train_file_*.h5")
        inputValFiles = glob.glob("/bigdata/shared/JetImages/converted/rotation_224_150p_v1/val_file_*.h5")

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
                torch.save(mymodel.state_dict(), "%s/IN%s_%s_bestmodel.params" %(loc, '_sumO' if mymodel.sum_O else '',param_string))
            else:
                print("Stale epoch")
                stale_epochs += 1
                if stale_epochs>=patience:
                    print("Early Stopping at",i)
                    # the last model
                    torch.save(mymodel.state_dict(), "%s/IN%s_%s_lastmodel.params" %(loc, '_sumO' if mymodel.sum_O else '',param_string))
                    break
    with open("%s/IN%s_bestmodel_loss.txt" %(loc, '_sumO' if mymodel.sum_O else ''), "a") as myfile:
        myfile.write("%s %f %i\n"%(param_string.replace("_"," "), best_loss_val, i+1))
    return best_loss_val

# function to optimize model
def f(x):
    print(x)
    gnn = GraphNet(nParticles, len(labels), params, int(x[:,0]), int(x[:,1]), int(x[:,2]), 
                   int(x[:,3]),  int(x[:,4]),  int(x[:,5]), int(x[:,6]), verbose=True, sum_O=args_sumO)
    param_string = '%s_%s_%s_%s_%s_%s_%s'%(int(x[:,0]), int(x[:,1]), int(x[:,2]), int(x[:,3]),  int(x[:,4]),  int(x[:,5]), int(x[:,6]))
    val_loss = model_evaluate(gnn, param_string)
    print("LOSS: %f" %val_loss)
    return val_loss

# run optimization
with open("%s/IN%s_bestmodel_loss.txt" %(loc, '_sumO' if args_sumO else ''), "w") as myfile:
    myfile.write('hidden De Do fr_activation fo_activation fc_activation optimizer best_val_loss n_epochs\n')
opt_model = GPyOpt.methods.BayesianOptimization(f=f, domain=bounds)
opt_model.run_optimization(max_iter=n_iter)

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

                               
with open("%s/IN%s_bestmodel_loss.txt" %(loc, '_sumO' if args_sumO else ''), "a") as myfile:
    myfile.write('hidden De Do fr_activation fo_activation fc_activation optimizer best_val_loss\n')
    myfile.write('%s %s %s %s %s %s %s %f\n'%(opt_model.x_opt[0],
                                              opt_model.x_opt[1],
                                              opt_model.x_opt[2],
                                              opt_model.x_opt[3],
                                              opt_model.x_opt[4],
                                              opt_model.x_opt[5],
                                              opt_model.x_opt[6],
                                              opt_model.fx_opt))
