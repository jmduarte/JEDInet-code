from torch.utils.data import Dataset, DataLoader
import h5py
from glob import glob
import numpy as np
import torch
from sklearn.utils import shuffle

class InEventLoader(Dataset):

    def check_data(self, file_names): 
        '''Count the number of events in each file and mark the threshold 
        boundaries between adjacent indices coming from 2 different files'''
        num_data = 0
        thresholds = [0]
        for in_file_name in file_names:
            # hardcoded !!!!
            f = h5py.File(in_file_name,'r')
            num_data += len(f.get(self.feature_name))
            f.close()
            thresholds.append(num_data)
        return (num_data, thresholds)

    def __init__(self, dir_name, nP, feature_name = 'jetConstituentList', label_name = 'jets', verbose = False):
        self.verbose = verbose
        self.feature_name = feature_name
        self.label_name = label_name
        self.file_names = glob("%s/*_%ip_*h5" %(dir_name, nP))
        random.shuffle(self.file_names)
        self.num_data, self.thresholds = self.check_data(self.file_names)
        self.file_index = 0
        self.h5_file = h5py.File(self.file_names[self.file_index], "r")
        self.X = np.array(self.h5_file.get(self.feature_name))
        self.Y = np.array(self.h5_file.get(self.label_name))[0:,-6:-1]
        self.X = np.swapaxes(self.X, 1, 2)
        self.Y = np.argmax(self.Y, axis=1)
        self.X = torch.FloatTensor(self.X)
        self.Y = torch.LongTensor(self.Y)

    def __init__(self, file_names, nP, feature_name = 'jetConstituentList', label_name = 'jets', verbose = False):
        self.verbose = verbose
        self.feature_name = feature_name
        self.label_name = label_name
        self.file_names = file_names
        self.num_data, self.thresholds = self.check_data(self.file_names)
        self.file_index = 0
        self.h5_file = h5py.File(self.file_names[self.file_index], "r")
        self.X = np.array(self.h5_file.get(self.feature_name))
        self.Y = np.array(self.h5_file.get(self.label_name))[0:,-6:-1]
        self.X = np.swapaxes(self.X, 1, 2)
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
        file_index = next(i for i,v in enumerate(self.thresholds) if v > idx)
        file_index -= 1
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
            self.h5_file.close() 
            if self.verbose: print("Opening new file: %s" %self.file_names[self.file_index])
            self.h5_file = h5py.File(self.file_names[self.file_index], 'r' )
            #print("LOAD_DATA")
            #h5_file = h5py.File( in_file_name, 'r' )
            self.X = np.array(self.h5_file.get(self.feature_name))
            self.Y = np.array(self.h5_file.get(self.label_name))[0:,-6:-1]
            self.X, self.Y = shuffle(self.X, self.Y)
            #h5_file.close()
            self.X = np.swapaxes(self.X, 1, 2)
            self.Y = np.argmax(self.Y, axis=1)
            self.X = torch.FloatTensor(self.X)
            self.Y = torch.LongTensor(self.Y)            
            self.file_index = file_index
        return {'jetConstituentList': self.X[event_index,:,:], 'jets': self.Y[event_index]}
    
if __name__=='__main__':
               
    import random


    labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
    params = ['j1_px', 'j1_py' , 'j1_pz' , 'j1_e' , 'j1_erel' , 'j1_pt' , 'j1_ptrel', 'j1_eta' , 'j1_etarel' , 
              'j1_etarot' , 'j1_phi' , 'j1_phirel' , 'j1_phirot', 'j1_deltaR' , 'j1_costheta' , 'j1_costhetarel']
    nParticles = 100
    inputTrainFiles = glob("/bigdata/shared/hls-fml/NEWDATA/jetImage*_%sp*.h5" %nParticles)
    inputValFiles = glob("/bigdata/shared/hls-fml/NEWDATA/VALIDATION/jetImage*_%sp*.h5" %nParticles)        
    batch_size = 1000
    
    train_set = InEventLoader(file_names=inputTrainFiles, nP=nParticles,
                              feature_name ='jetConstituentList',label_name = 'jets', verbose=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_set = InEventLoader(file_names=inputValFiles, nP=nParticles,
                            feature_name ='jetConstituentList',label_name = 'jets', verbose=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    sums = np.zeros((len(labels)))
    import tqdm
    for t in tqdm.tqdm(train_loader,total=len(train_set)/batch_size):
        a = t['jets'].cpu().numpy()
        b = np.zeros((a.shape[0], len(labels)))
        b[np.arange(len(a)), a] = 1
        sums += np.sum(b,axis=0)
    print('training', sums)
    sums_val = np.zeros((len(labels)))
    for t in tqdm.tqdm(val_loader,total=len(val_set)/batch_size):
        a = t['jets'].cpu().numpy()
        b = np.zeros((a.shape[0], len(labels)))
        b[np.arange(len(a)), a] = 1
        sums_val += np.sum(b,axis=0)
    print('validation', sums_val)
    sums_tot = sums+sums_val
    print('total', sums_tot)

