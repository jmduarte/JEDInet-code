import setGPU
import os
import numpy as np
import h5py
import glob
import itertools
import sys
from sklearn.utils import shuffle
from sklearn import metrics

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


#loc='IN_Top_Big_%s'%(sys.argv[1])
loc='IN_Top_Hyper_Old_%s'%(sys.argv[1])
#loc='IN_Top_Hyper_New_%s'%(sys.argv[1])
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

import glob
import os

if os.path.isdir('/bigdata/shared'):
    inputTestFiles = glob.glob("/bigdata/shared/JetImages/converted/rotation_224_150p_v1/test_file_*.h5")

# hidden, De, Do, fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0
mymodel = GraphNet(nParticles, len(labels), params, int(x[0]), int(x[1]), int(x[2]), 
                   fr_activation=int(x[3]),  fo_activation=int(x[4]), fc_activation=int(x[5]), optimizer=int(x[6]), verbose=True, 
                   sum_O=args_sumO)

mymodel.load_state_dict(torch.load("%s/IN%s_bestmodel.params" %(loc, '_sumO' if mymodel.sum_O else '')))

loss = nn.CrossEntropyLoss(reduction='mean')

test_set = InEventLoaderTop(file_names=inputTestFiles, nP=nParticles,
                             feature_names = params,label_name = 'label', verbose=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

nBatches_per_test_epoch = len(test_set)/batch_size
print("nBatches_per_test_epoch: %i" %nBatches_per_test_epoch)

# testing
t = tqdm.tqdm(enumerate(test_loader), total = len(test_set)/batch_size)
mymodel.eval()
if not os.path.isfile("%s/testing%s.h5" %(loc, '_sumO' if mymodel.sum_O else '')):
    with torch.no_grad():
        out_tests = []
        targets = []
        for batch_idx, mydict in t:
            data = mydict['jetConstituentList']
            target = mydict['jets']
            if args_cuda:
                data, target = data.cuda(), target.cuda()
            out_test = mymodel(data)
            out_tests  += [out_test]
            targets  += [target]
            l_test = loss(out_test, target)
            acc_test = accuracy(out_test, target)
            loss_test_item = l_test.cpu().data.numpy()
            t.set_description("test batch loss = %.5f, acc = %.5f" % (loss_test_item, acc_test))
            t.refresh() # to show immediately the update
        targets = torch.cat(targets,0)
        out_tests = torch.cat(out_tests,0)
        acc_test = accuracy(out_tests, targets)
        loss_test = loss(out_tests, targets)
        if mymodel.verbose: 
            print("Testing Loss: %f" %loss_test)
            print("Testing Acc: %f" %acc_test)

        targets = targets.cpu().data.numpy()
        out_tests = out_tests.cpu().data.numpy()
        with h5py.File("%s/testing%s.h5" %(loc, '_sumO' if mymodel.sum_O else ''), "w") as f:
            f.create_dataset('out_test', data=out_tests, compression='gzip')
            f.create_dataset('target_test', data=targets, compression='gzip')
else:
    with h5py.File("%s/testing%s.h5" %(loc, '_sumO' if mymodel.sum_O else ''), "r") as f:
        out_tests = f['out_test'][()]
        targets = f['target_test'][()]
               

from scipy.special import softmax
softmax_out = softmax(out_tests,axis=1)

fpr_test, tpr_test, thresholds = metrics.roc_curve(1-targets, softmax_out[:,0])
# Find the true positive rate of 30% and 1 over the false positive rate at tpr = 30%.
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

idx_t = find_nearest(tpr_test,0.3)
accuracy = metrics.accuracy_score(targets, np.argmax(out_tests,axis=1))
auc_test = metrics.auc(fpr_test, tpr_test)

print('accuracy', accuracy*100.)
print('auc', auc_test*100.)
print('1/eB@eS=.3', 1./fpr_test[idx_t])
import matplotlib.pyplot as plt
plt.style.use('sonic.mplstyle')
plt.figure(figsize=(7,5))
plt.plot(tpr_test, fpr_test, label=r'JEDI-net%s: acc = %.1f%%, AUC = %.1f%%, $1/\epsilon_{B}$ = %.0f'%(' $\Sigma O$' if mymodel.sum_O else '',accuracy*100., auc_test*100.,  1./fpr_test[idx_t]))
plt.semilogy()
plt.xlabel("Signal efficiency",fontsize='x-large')
plt.ylabel("Background efficiency",fontsize='x-large')
plt.ylim(0.0001,1)
plt.xlim(0,1)
plt.grid(True)
plt.legend(loc='upper left',fontsize=11.8)
plt.tight_layout()
plt.savefig('%s/ROC%s.pdf'%(loc, '_sumO' if mymodel.sum_O else ''))

plt.figure(figsize=(7,5))
plt.hist(softmax_out[:,0], weights=1-targets, bins = np.linspace(0, 1, 41), label='signal',alpha=0.5)
plt.hist(softmax_out[:,0], weights=targets, bins = np.linspace(0, 1, 41), label='background',alpha=0.5)
plt.legend(loc='upper left',fontsize=11.8)
plt.tight_layout()
plt.savefig('%s/dist%s.pdf'%(loc,  '_sumO' if mymodel.sum_O else ''))
