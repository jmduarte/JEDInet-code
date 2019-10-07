from torch.utils.data import Dataset, DataLoader
import h5py
import tables
from glob import glob
import numpy as np
import torch
from sklearn.utils import shuffle

class InEventLoaderTop(Dataset):

    def check_data(self, file_names): 
        '''Count the number of events in each file and mark the threshold 
        boundaries between adjacent indices coming from 2 different files'''
        num_data = 0
        thresholds = [0]
        for in_file_name in file_names:
            # hardcoded !!!!
            f = tables.open_file(in_file_name,'r')
            num_data += np.array(getattr(f.root,self.label_name)).shape[0]
            f.close()
            thresholds.append(num_data)
        return (num_data, thresholds)

    def __init__(self, file_names, nP, feature_names = ['part_px','part_py','part_pz'], label_name = 'label', verbose = False):
        self.verbose = verbose
        self.feature_names = feature_names
        self.label_name = label_name
        self.file_names = file_names
        self.num_data, self.thresholds = self.check_data(self.file_names)
        self.file_index = 0
        self.h5_file = tables.open_file(self.file_names[self.file_index], "r")
        lists = []
        for feature_name in self.feature_names:
            if feature_name=='part_costheta':
                eta = np.array(self.h5_file.root.part_eta)
                f = np.cos(2.*np.arctan(np.exp(eta)))
            elif feature_name=='part_costhetarel':
                eta_rot = np.array(self.h5_file.root.part_eta_rot)
                f = np.cos(2.*np.arctan(np.exp(eta_rot)))
            else:
                f = np.array(getattr(self.h5_file.root,feature_name))
            f = f.reshape(f.shape[0],f.shape[1],1)
            lists.append(f)
        self.X = np.concatenate(lists,axis=-1)
        self.Y = np.array(getattr(self.h5_file.root,label_name))
        self.X = np.swapaxes(self.X, 1, 2)
        self.X, self.Y = shuffle(self.X, self.Y)
        self.Y = np.argmax(self.Y, axis=1)
        self.X = torch.FloatTensor(self.X)
        self.Y = torch.LongTensor(self.Y)

    def load_data(self, h5_file):
        """Loads numpy arrays from H5 file.
            If the features/labels groups contain more than one dataset,
            we load them all, alphabetically by key."""

        return X,Y

    def get_data(self, data, idx):
        """Input: a numpy array or list of numpy arrays.
            Gets elements at idx for each array"""
        return data[idx]

    def get_index(self, idx):
        """Translate the global index (idx) into local indexes,
        including file index and event index of that file"""
        for i in range(len(self.thresholds)-1):
            if idx >= self.thresholds[i] and idx < self.thresholds[i+1]: 
                file_index = i
                break
        event_index = idx - self.thresholds[file_index]
        return file_index, event_index

    def get_thresholds(self):
        return self.thresholds

    # Below are the two functions you are required to define
    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        #print("LOAD BATCH")
        file_index, event_index = self.get_index(idx)
        if file_index != self.file_index:
            self.file_index = file_index
            self.h5_file.close() 
            if self.verbose: 
                print("Opening new file: %s" %self.file_names[self.file_index])
            self.h5_file = tables.open_file(self.file_names[self.file_index], 'r' )
            #print("LOAD_DATA")
            #h5_file = h5py.File( in_file_name, 'r' )
            lists = []
            for feature_name in self.feature_names:
                if feature_name=='part_costheta':
                    eta = np.array(self.h5_file.root.part_eta)
                    f = np.cos(2.*np.arctan(np.exp(eta)))
                elif feature_name=='part_costhetarel':
                    eta_rot = np.array(self.h5_file.root.part_eta_rot)
                    f = np.cos(2.*np.arctan(np.exp(eta_rot)))
                else:
                    f = np.array(getattr(self.h5_file.root,feature_name))
                f = f.reshape(f.shape[0],f.shape[1],1)
                lists.append(f)
            self.X = np.concatenate(lists,axis=-1)
            self.Y = np.array(getattr(self.h5_file.root,self.label_name))
            self.X, self.Y = shuffle(self.X, self.Y)
            #h5_file.close()
            self.X = np.swapaxes(self.X, 1, 2)
            self.Y = np.argmax(self.Y, axis=1)
            self.X = torch.FloatTensor(self.X)
            self.Y = torch.LongTensor(self.Y)     
        return {'jetConstituentList': self.X[event_index,:,:], 'jets': self.Y[event_index]}
    
if __name__=='__main__':
               
    import random

    labels = ['isTop','isQCD']
    params = ['part_px', 'part_py' , 'part_pz' , 
              'part_energy' , 'part_erel' , 'part_pt' , 'part_ptrel', 
              'part_eta' , 'part_etarel' ,
              'part_eta_rot' , 
              'part_phi' , 'part_phirel' , 
              'part_phi_rot', 'part_deltaR',
              'part_costheta' , 'part_costhetarel']

    inputTrainFiles = glob("/bigdata/shared/JetImages/converted/rotation_224_v1/train_file_*.h5")
    inputValFiles = glob("/bigdata/shared/JetImages/converted/rotation_224_v1/val_file_*.h5")
    batch_size = 1024
    nParticles = 100
    
    train_set = InEventLoaderTop(file_names=inputTrainFiles, nP=nParticles,
                              feature_names = params,label_name = 'label', verbose=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_set = InEventLoaderTop(file_names=inputValFiles, nP=nParticles,
                            feature_names = params, label_name = 'label', verbose=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    for t in train_loader:
        print(t['jetConstituentList'].shape)
        print(t['jets'].shape)
        #print(t['jetConstituentList'][0])
        #print(t['jets'][0])

