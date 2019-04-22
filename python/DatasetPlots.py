#!/usr/bin/env python
# coding: utf-8
import h5py
import glob
import numpy as np

# need afs password to read from eos
onEOS = False
import os
import getpass
if onEOS: os.system("echo %s| kinit" %getpass.getpass())
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
plt.style.use('sonic.mplstyle')
#get_ipython().magic(u'matplotlib inline')

# # Read Dataset: HLF
### TO BE CHANGED
fileIN = "../data/jetImage_30.h5"
print(fileIN)
f = h5py.File(fileIN)
data = np.array(f.get("jets"))
data_q = data[data[:,-6]==1]
data_g = data[data[:,-5]==1]
data_W = data[data[:,-4]==1]
data_Z = data[data[:,-3]==1]
data_t = data[data[:,-2]==1]
### select only relevant features
data_q = data_q[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]]
data_g = data_g[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]]
data_W = data_W[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]]
data_Z = data_Z[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]]
data_t = data_t[:,[12, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 52]]
####
print(data_q.shape, data_g.shape, data_W.shape, data_Z.shape, data_t.shape)
print(data[:,-6])


labels = [r'$\sum z\log(z)$', r'$C_{1}^{0}$', r'$C_{1}^{1}$', r'$C_{1}^{2}$',
          r'$C_{2}^{1}$', r'$C_{2}^{2}$',
          r'$D_{2}^{1}$', r'$D_{2}^{2}$',
          r'$D_{2}^{(1,1)}$', r'$D_{2}^{(1,2)}$',
          r'$M_{2}^{1}$', r'$M_{2}^{2}$',
          r'$N_{2}^{1}$', r'$N_{2}^{2}$', r'$m_{mMDT}$ [GeV]', r'Multiplicity']

minEntries = min(data_q.shape[0],data_g.shape[0],data_W.shape[0],
                data_Z.shape[0],data_t.shape[0])

for i in range(16):
    plt.hist(data_q[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(data_g[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(data_W[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(data_Z[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(data_t[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.xlabel(labels[i], fontsize=15)
    plt.ylabel('Prob. Density (a.u.)', fontsize=15)
    #handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
    labelCat= ["quark","gluon", "W", "Z", "top"]
    plt.legend(labelCat, fontsize=12, frameon=False)  
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    #plt.show()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.draw()
    plt.savefig('fig_%i.png' %i, dpi=250)
    plt.savefig('fig_%i.pdf' %i, dpi=250)
    plt.close()

#### LOG
minEntries = min(data_q.shape[0],data_g.shape[0],data_W.shape[0],
                data_Z.shape[0],data_t.shape[0])
for i in range(16):
    plt.hist(data_q[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(data_g[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(data_W[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(data_Z[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(data_t[:minEntries,i], 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.xlabel(labels[i], fontsize=15)
    plt.ylabel('Prob. Density (a.u.)', fontsize=15)
    #handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
    labelCat= ["quark","gluon", "W", "Z", "top"]
    plt.yscale('log', nonposy='clip')
    plt.legend(labelCat, fontsize=12, frameon=False)  
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.draw()
    plt.savefig('fig_%i_log.png' %i, dpi=250)
    plt.savefig('fig_%i_log.pdf' %i, dpi=250)
    plt.close()

# # plot average images
from matplotlib.colors import LogNorm
image = np.array(f.get('jetImage'))
image_q = image[data[:,-6]==1]
image_g = image[data[:,-5]==1]
image_W = image[data[:,-4]==1]
image_Z = image[data[:,-3]==1]
image_t = image[data[:,-2]==1]
images = [image_q, image_g, image_W, image_Z, image_t]

for i in range(len(images)):
    SUM_Image = np.sum(images[i], axis = 0)
    plt.imshow(SUM_Image/float(images[i].shape[0]), origin='lower',norm=LogNorm(vmin=0.01))
    plt.colorbar()
    plt.title(labelCat[i], fontsize=20)
    plt.xlabel(r"$\Delta\eta$ cell", fontsize=15)
    plt.ylabel(r"$\Delta\phi$ cell", fontsize=15)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.draw()
    plt.savefig('image_%i_log.png' %i, dpi=250, bbox_inches='tight')
    plt.savefig('image_%i_log.pdf' %i, dpi=250, bbox_inches='tight')
    plt.close()


# # plot single-jet images
image_q = image_q[10,:,:]
image_g = image_g[10,:,:]
image_W = image_W[10,:,:]
image_Z = image_Z[10,:,:]
image_t = image_t[10,:,:]
images = [image_q, image_g, image_W, image_Z, image_t]
for i in range(len(images)):
    #SUM_Image = np.sum(images[i], axis = 0)
    plt.imshow(images[i], origin='lower',norm=LogNorm(vmin=0.01))
    plt.colorbar()
    plt.title(labelCat[i], fontsize=20)
    plt.xlabel(r"$\Delta\eta$ cell", fontsize=15)
    plt.ylabel(r"$\Delta\phi$ cell", fontsize=15)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.draw()
    plt.savefig('imageOneJet_%i_log.png' %i, dpi=250, bbox_inches='tight')
    plt.savefig('imageOneJet_%i_log.pdf' %i, dpi=250, bbox_inches='tight')
    plt.close()


# # plot constituent features
listP = np.array(f.get('jetConstituentList'))
list_q = listP[data[:,-6]==1]
list_g = listP[data[:,-5]==1]
list_W = listP[data[:,-4]==1]
list_Z = listP[data[:,-3]==1]
list_t = listP[data[:,-2]==1]
print(list_q.shape)
minEntries = min(list_q.shape[0],list_g.shape[0],list_W.shape[0],
                list_Z.shape[0],list_t.shape[0])
list_q = list_q[:minEntries,:,:]
list_g = list_g[:minEntries,:,:]
list_W = list_W[:minEntries,:,:]
list_Z = list_Z[:minEntries,:,:]
list_t = list_t[:minEntries,:,:]
print(list_q.shape)

d2 = f.get("particleFeatureNames")
print(d2[:,])

featureNames = [r'$p_{x}$ [GeV]', r'$p_{y}$ [GeV]', r'$p_{z}$ [GeV]', r'$E$ [GeV]',
                    r'Relative $E$', r'$p_{T}$ [GeV]', r'Relative $p_{T}$',
                    r'$\eta$', r'Relative $\eta$', r'Rotated $\eta$', r'$\phi$', r'Relative $\phi$',
                    r'Rotated $\phi$', r'$\Delta R$',r'$\cos\theta$', r'Relative $\cos \theta$']

for i in range(16):
    plt.hist(np.ndarray.flatten(list_q[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(np.ndarray.flatten(list_g[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(np.ndarray.flatten(list_W[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(np.ndarray.flatten(list_Z[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(np.ndarray.flatten(list_t[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    print(i,featureNames[i])
    plt.xlabel(featureNames[i], fontsize=15)
    plt.ylabel('Prob. Density (a.u.)', fontsize=15)
    #handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
    labelCat= ["quark","gluon", "W", "Z", "top"]
    plt.legend(labelCat, fontsize=12, frameon=False)  
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    #plt.show()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.draw()
    plt.savefig('const_fig_%i.png' %i, dpi=250)
    plt.savefig('const_fig_%i.pdf' %i, dpi=250)
    plt.close()

# now in log scale
for i in range(16):
    plt.hist(np.ndarray.flatten(list_q[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(np.ndarray.flatten(list_g[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(np.ndarray.flatten(list_W[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(np.ndarray.flatten(list_Z[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.hist(np.ndarray.flatten(list_t[:,:,i]), 50, density=True, histtype='step', fill=False, linewidth=1.5)
    plt.xlabel(featureNames[i], fontsize=15)
    plt.ylabel('Prob. Density (a.u.)', fontsize=15)
    #handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
    labelCat= ["quark","gluon", "W", "Z", "top"]
    plt.legend(labelCat, fontsize=12, frameon=False)  
    plt.yscale('log', nonposy='clip')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.draw()
    plt.savefig('const_log_fig_%i.png' %i, dpi=250)
    plt.savefig('const_log_fig_%i.pdf' %i, dpi=250)
    plt.close()

