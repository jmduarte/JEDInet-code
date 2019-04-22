
# coding: utf-8

# In[1]:


args_cuda = 0
args_gpu = False

if args_gpu: 
    import setGPU
    args_cuda = 1
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

#get_ipython().magic(u'matplotlib inline')
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.style.use('sonic.mplstyle')


# define the pytorch model
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
        if args_cuda: 
            self.fr1 = nn.Linear(2 * self.P + self.Dr, hidden).cuda()
            self.fr2 = nn.Linear(hidden, int(hidden/2)).cuda()
            self.fr3 = nn.Linear(int(hidden/2), self.De).cuda()
            self.fo1 = nn.Linear(self.P + self.Dx + self.De, hidden).cuda()
            self.fo2 = nn.Linear(hidden, int(hidden/2)).cuda()
            self.fo3 = nn.Linear(int(hidden/2), self.Do).cuda()
            self.fc1 = nn.Linear(self.Do * self.N, hidden).cuda()
            self.fc2 = nn.Linear(hidden, int(hidden/2)).cuda()
            self.fc3 = nn.Linear(int(hidden/2), self.n_targets).cuda()
        else:
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
        if args_cuda:
            self.Rr = Variable(self.Rr).cuda()
            self.Rs = Variable(self.Rs).cuda()
        else:
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
        #del O
        #N = nn.functional.relu(self.fc3(N))                                                                            
        N = self.fc3(N)
        return N, O

    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L                                                       
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])


# optimal parameters
nParticles = 100
x = []
x.append(50) # hinned nodes                                                                                             
x.append(12) # De                                                                                                       
x.append(4) # Do                                                                                                        
x.append(2) # fr_activation_index                                                                                       
x.append(0) # fo_activation_index                                                                                       
x.append(0) # fc_activation_index                                                                                       
x.append(0) # optmizer_index                


labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
classes = ['gluons', 'quarks', 'W','Z', 'top']
params = ['j1_px', 'j1_py' , 'j1_pz' , 'j1_e' , 'j1_erel' , 'j1_pt' , 'j1_ptrel', 'j1_eta' , 'j1_etarel' ,
          'j1_etarot' , 'j1_phi' , 'j1_phirel' , 'j1_phirot', 'j1_deltaR' , 'j1_costheta' , 'j1_costhetarel']


val_split = 0.3
batch_size = 100
n_epochs = 100
patience = 10


# In[8]:


import glob
inputValFiles = glob.glob("../data/jetImage*_%sp*.h5" %nParticles)


# In[9]:


mymodel = GraphNet(nParticles, len(labels), params, int(x[0]), int(x[1]), int(x[2]),
                   int(x[3]),  int(x[4]),  int(x[5]), int(x[6]), 0)
mymodel.load_state_dict(torch.load("../models//IN_100.params", map_location='cpu'))
mymodel.eval()


# In[10]:


# HLF  in classifier
myHLFlist = [12, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 52]
namesY = ['zlogz', 'c1_b0_mmdt', 'c1_b1_mmdt', 'c1_b2_mmdt', 'c2_b1_mmdt', 'c2_b2_mmdt', 
          'd2_b1_mmdt', 'd2_b2_mmdt','d2_a1_b2_mmdt', 'm2_b1_mmdt', 
          'm2_b2_mmdt', 'n2_b1_mmdt', 'n2_b2_mmdt', 'mass_mmdt', 'multiplicity']
labelsY = [r'$\sum z\log(z)$', r'$C_{1}^{0}$', r'$C_{1}^{1}$', r'$C_{1}^{2}$',
          r'$C_{2}^{1}$', r'$C_{2}^{2}$',
         r'$D_{2}^{1}$',r'$D_{2}^{2}$',
          r'$D_{2}^{(1,2)}$', 
         r'$M_{2}^{1}$', r'$M_{2}^{2}$',
         r'$N_{2}^{1}$', r'$N_{2}^{2}$', r'$m_{mMDT}$', r'Multiplicity']

# other HLF: all
myHLFlist.extend([4,26,7,29,
             5,27,8,30,
             6,28,9,31])
namesY.extend(['j_tau1_b1', 'j_tau1_b1_mmdt','j_tau1_b2',  'j_tau1_b2_mmdt', 
          'j_tau2_b1', 'j_tau2_b1_mmdt','j_tau2_b2',  'j_tau2_b2_mmdt', 
          'j_tau3_b1', 'j_tau3_b1_mmdt','j_tau3_b2',  'j_tau3_b2_mmdt'])
labelsY.extend([r'$\tau_1^{(\beta=1)}$',r'$\tau_1^{(\beta=1,mMDT)}$',r'$\tau_1^{(\beta=2)}$',r'$\tau_1^{(\beta=2,mMDT)}$',
           r'$\tau_2^{(\beta=1)}$',r'$\tau_2^{(\beta=1,mMDT)}$',r'$\tau_2^{(\beta=2)}$',r'$\tau_2^{(\beta=2,mMDT)}$',
           r'$\tau_3^{(\beta=1)}$',r'$\tau_3^{(\beta=1,mMDT)}$',r'$\tau_3^{(\beta=2)}$',r'$\tau_3^{(\beta=2,mMDT)}$'])


#read datasets
X = np.array([])                                                                                                                                      
Y = np.array([]) 
Y_hlf = np.array([]) 
for fileIN in inputValFiles:
    if X.shape[0] >10000: continue
    f = h5py.File(fileIN, 'r')                                                                                                                        
    myFeatures = np.array(f.get('jetConstituentList'))                                                                                                
    myTarget = np.array(f.get('jets')[0:,-6:-1])
    myHLF = np.array(f.get('jets'))
    X = np.concatenate([X,myFeatures], axis = 0) if X.size else myFeatures                                                                            
    Y = np.concatenate([Y,myTarget], axis = 0) if Y.size else myTarget                                                                                
    Y_hlf = np.concatenate([Y_hlf,myHLF], axis = 0) if Y_hlf.size else myHLF                                                                                
    print(X.shape, Y.shape, Y_hlf.shape)


# pre-processing                                                                                                                                
X, Y, Y_hlf = shuffle(X, Y, Y_hlf, random_state=1)
X = np.swapaxes(X, 1, 2)
#Y = np.argmax(Y, axis=1)
X = torch.FloatTensor(X)
Y = torch.FloatTensor(Y)
Y_hlf = torch.FloatTensor(Y_hlf)

# extract the O matrix and the category output [TBF]
predict_test = []
lst = []
Otot = []
for j in torch.split(X, batch_size):
    a, myO = mymodel(j)
    a = a.cpu().data.numpy()
    myO = torch.sum(myO, dim=1)
    myO = myO.cpu().data.numpy()
    # sum over particles
    lst.append(a)
    Otot.append(myO)
    
predicted = Variable(torch.FloatTensor(np.concatenate(lst)))
predicted = torch.nn.functional.softmax(predicted, dim=1)
predict_test = predicted.data.numpy()

O_predicted = Variable(torch.FloatTensor(np.concatenate(Otot)))
O_predicted_test = O_predicted.data.numpy()

# ROC CURVE
import pandas as pd
from sklearn.metrics import roc_curve, auc
df = pd.DataFrame()
fpr = {}
tpr = {}
auc1 = {}
plt.figure()
for i, label in enumerate(labels):
        df[label] = Y[:,i]
        df[label + '_pred'] = predict_test[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])

        auc1[label] = auc(fpr[label], tpr[label])
        print('%s tagger, auc = %.1f%%' %(label,auc1[label]*100.))
        #plt.plot(tpr[label],fpr[label],label='%s tagger, auc = %.1f%%' %(label,auc1[label]*100.))
        plt.plot(tpr[label],fpr[label])
plt.semilogy()
plt.xlabel("sig. efficiency")
plt.ylabel("bkg. mistag rate")
plt.ylim(0.000001,1)
plt.grid(True)
plt.legend(loc='upper left')
#plt.savefig('%s/ROC.pdf'%(options.outputDir))
plt.show()

print(O_predicted_test.shape)
print(Y_hlf.shape)

mask = Y[:,0].data.numpy()
mask = mask >0
print(mask)


myrho = np.zeros([4,len(labelsY)])
from matplotlib.colors import LogNorm
for i in range(4):
    xlabel = "$\overline{O_{%i}}$" %i
    myO_predicted_test = O_predicted_test[:,i]/100.
    for j in range(len(labelsY)):
        myY_hlf = Y_hlf.data.numpy()[:,myHLFlist[j]]
        ylabel = labelsY[j]
        plt.hist2d(myO_predicted_test[mask], myY_hlf[mask], bins=30, norm=LogNorm())
        plt.xlabel(xlabel, fontsize=15)
        plt.ylabel(ylabel, fontsize=15)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        rho = np.corrcoef(myO_predicted_test[mask], myY_hlf[mask])
        print(rho[0,1])
        myrho[i,j] = rho[0,1]
        #plt.legend(xlabel, '$rho_{corr} = %f$' %rho[1,1])
        plt.show()
        #plt.draw()
        #plt.savefig('O%i_%s.png' %(i, namesY[j]), dpi=250)
    
# drow correlation plot
fig, ax = plt.subplots(figsize=(8, 2*len(labelsY)))

#fig.figure(figsize=(8, 24))
#im = ax.imshow(myrho, origin='lower',norm=LogNorm(vmin=0.01), cmap='Blues')
im = ax.imshow(myrho, origin='lower', cmap='bwr', vmin=-1, vmax=1)

# We want to show all ticks...
ax.set_xticks(np.arange(len(labelsY)))
ax.set_yticks(np.arange(4))
# ... and label them with the respective list entries
ax.set_yticklabels([r"$\overline{O}_0$", r"$\overline{O}_1$", r"$\overline{O}_2$", r"$\overline{O}_3$"], fontsize=15)
ax.set_xticklabels(labelsY, fontsize=15)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
# Loop over data dimensions and create text annotations.
for i in range(4):
    for j in range(len(labelsY)):
        text = plt.text(j, i, "%.2f" %myrho[i, j], ha="center", va="center", color="black")
        
# Create colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

#cbar = ax.figure.colorbar(im, ax=ax, cmap="binary")
cbar = ax.figure.colorbar(im, cax=cax, cmap='Blues')
cbar.ax.set_ylabel("Linear correlation coeff.", rotation=-90, va="bottom", fontsize=15)

for i_Tau in range(len(labelsY)):
    for i_O in range(0,4):
        ylabel = labelsY[i_Tau]
        for iClass in range(5):
            print(i_O, namesY[i_Tau], classes[iClass])
            mask = Y[:,iClass].data.numpy()
            mask = mask >0
            # make 2D plot
            xlabel = "$\overline{O_{%i}}$" %i_O
            myO_predicted_test = O_predicted_test[:,i_O]/100.
            myY_hlf = Y_hlf.data.numpy()[:,myHLFlist[i_Tau]]
            rho = np.corrcoef(myO_predicted_test[mask], myY_hlf[mask])
            print(rho[0,1])
            plt.figure()
            plt.hist2d(myO_predicted_test[mask], myY_hlf[mask], bins=30, norm=LogNorm())
            plt.xlabel(xlabel, fontsize=15)
            plt.ylabel(ylabel, fontsize=15)
            plt.title(r"%s ($\rho = %.2f$)" %(classes[iClass], rho[0,1]), fontsize=15)
            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            myrho[i,j] = rho[0,1]
            #plt.legend(xlabel, '$rho_{corr} = %f$' %rho[1,1])
            #plt.show()
            plt.draw()
            plt.savefig('O%i_%s_orr_%s.png' %(i_O, namesY[i_Tau], classes[iClass]), dpi=250)
            plt.savefig('O%i_%s_corr_%s.pdf' %(i_O, namesY[i_Tau], classes[iClass]), dpi=250)
            plt.close()

