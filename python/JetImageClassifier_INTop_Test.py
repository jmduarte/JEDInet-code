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

args_cuda = bool(sys.argv[2])
args_sumO = bool(int(sys.argv[3])) if len(sys.argv)>3 else False


loc='IN_Top_Big_%s'%(sys.argv[1])
import os
os.system('mkdir -p %s'%loc)

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
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fr3 = nn.Linear(int(self.hidden/2), self.De).cuda()
        self.fo1 = nn.Linear(self.P + self.Dx + self.De, self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fo3 = nn.Linear(int(self.hidden/2), self.Do).cuda()
        if self.sum_O:
            self.fc1 = nn.Linear(self.Do *1, self.hidden).cuda()
        else:
            self.fc1 = nn.Linear(self.Do * self.N, self.hidden).cuda()
        self.fc2 = nn.Linear(self.hidden, int(self.hidden/2)).cuda()
        self.fc3 = nn.Linear(int(self.hidden/2), self.n_targets).cuda()

    def assign_matrices(self):
        self.Rr = torch.zeros(self.N, self.Nr)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = self.Rr.cuda()
        self.Rs = self.Rs.cuda()

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
    #30 : [10.,  8., 12.,  0.,  1.,  0.,  0.],
    #50 : [50., 14., 14.,  0.,  0.,  2.,  0.],
    #100 : [30., 10.,  8.,  2.,  1.,  1.,  0.],
    #150 : []
    ## 50 epochs, 10 patience., 10 iterations
    30 : [50., 12.,  6.,  0.,  2.,  2.,  0.], #optimized loss: 0.6316463625210308
    50 : [50., 12., 14.,  1.,  2.,  1.,  0.], ##50 epochs: optimized loss: 0.5712810956438387
    100 :     [10.,  8.,  8.,  0.,  1.,  1.,  1.], #LOSS: 0.618831
    #150 : [50., 14.,  6.,  2.,  2.,  0.,  0.]#LOSS: 0.554133
    150 : [128., 64.,  64.,  0.,  0.,  0.,  0.]
}
sumO_best_perf = {
    # hidden, De, Do, fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0
    #30 : [50.,  4.,  4.,  2.,  0.,  2.,  0.],
    #50 : [50.,  8., 14.,  2.,  0.,  2.,  0.],
    #100: [40., 10., 12.,  2.,  2.,  2.,  0.],
    #150 : [40., 10., 12.,  2.,  0.,  2.,  0.]
    30 : [6., 8., 6., 0., 1., 1., 0.], #optimized loss: 0.8398462489357698
    50 : [50., 12., 14., 0.,  0.,  2.,  0.], #optimized loss: 0.5850381782526777
    100 : [30.,  4.,  4.,  2.,  0.,  2.,  0.], #optimized loss: 0.6234710748617857
    #150 :     [10.,  6.,  6.,  0.,  2.,  1.,  0.] # LOSS: 0.617842
    150 : [128., 64.,  64.,  0.,  0.,  0.,  0.]
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
patience = 10

import glob
import os

if os.path.isdir('/bigdata/shared'):
    inputTestFiles = glob.glob("/bigdata/shared/JetImages/converted/rotation_224_150p_v1/test_file_*.h5")

# hidden, De, Do, fr_activation=0, fo_activation=0, fc_activation=0, optimizer = 0
mymodel = GraphNet(nParticles, len(labels), params, int(x[0]), int(x[1]), int(x[2]), 
                   fr_activation=int(x[3]),  fo_activation=int(x[4]), fc_activation=int(x[5]), optimizer=int(x[6]), verbose=True)

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
               
fpr_test, tpr_test, thresholds = metrics.roc_curve(1-targets, out_tests[:,0])
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
