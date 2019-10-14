#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s')
import pandas as pd
import numpy as np
import numexpr as ne
import math
import tables
filters = tables.Filters(complevel=7, complib='blosc')

def rotate_and_reflect_element(x,y,w):
    rot_x = []
    rot_y = []
    theta = 0
    maxPt = -1
    for ix, iy, iw in zip(x, y, w):
        #print(ix)
        dR = np.sqrt(ix*ix+iy*iy)
        thisPt = iw
        if dR > 0.1 and thisPt > maxPt:
            maxPt = thisPt
            # rotation in eta-phi plane c.f  https://arxiv.org/abs/1407.5675 and https://arxiv.org/abs/1511.05190:
            # theta = -np.arctan2(iy,ix)-np.radians(90)
            # rotation by lorentz transformation c.f. https://arxiv.org/abs/1704.02124:
            px = iw*np.cos(iy)
            py = iw*np.sin(iy)
            pz = iw*np.sinh(ix)
            theta = np.arctan2(py,pz)+np.radians(90)
            
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
    for ix, iy, iw in zip(x, y, w):
        # rotation in eta-phi plane:
        #rot = R*np.matrix([[ix],[iy]])
        #rix, riy = rot[0,0], rot[1,0]
        # rotation by lorentz transformation
        px = iw*np.cos(iy)
        py = iw*np.sin(iy)
        pz = iw*np.sinh(ix)
        rot = R*np.matrix([[py],[pz]])
        px1 = px
        py1 = rot[0,0]
        pz1 = rot[1,0]
        iw1 = np.sqrt(px1*px1+py1*py1)
        rix, riy = np.arcsinh(pz1/iw1), np.arcsin(py1/iw1)
        rot_x.append(rix)
        rot_y.append(riy)
        
    # now reflect if leftSum > rightSum
    leftSum = 0
    rightSum = 0
    for ix, iy, iw in zip(x, y, w):
        if ix > 0: 
            rightSum += iw
        elif ix < 0:
            leftSum += iw
    if leftSum > rightSum:
        ref_x = [-1.*rix for rix in rot_x]
        ref_y = rot_y
    else:
        ref_x = rot_x
        ref_y = rot_y
    
    return np.array(ref_x), np.array(ref_y)

def rotate_and_reflect(x,y,w):
    rot_x = []
    rot_y = []
    theta = 0
    dR = np.sqrt(x*x+y*y)
    mask = dR > 0.1
    w_masked = np.multiply(mask.astype(np.int),w) # mask particles with dR<0.1
    maxPt_mindR = np.argmax(w_masked,axis=1) # find highest Pt one
    ind = np.unravel_index(np.argmax(w_masked, axis=1), w_masked.shape)
    px = w_masked[ind]*np.cos(y[ind])
    py = w_masked[ind]*np.sin(y[ind])
    pz = w_masked[ind]*np.sinh(x[ind])
    theta = np.arctan2(py,pz)+np.radians(90) 
    c, s = np.cos(theta), np.sin(theta)
    R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
#    R = np.array([c,-s],[s,c])
    for ix, iy, iw in zip(x, y, w):
        # rotation in eta-phi plane:
        #rot = R*np.matrix([[ix],[iy]])
        #rix, riy = rot[0,0], rot[1,0]
        # rotation by lorentz transformation
        px = iw*np.cos(iy)
        py = iw*np.sin(iy)
        pz = iw*np.sinh(ix)
        rot = R*np.matrix([[py],[pz]])
        px1 = px
        py1 = rot[0,0]
        pz1 = rot[1,0]
        iw1 = np.sqrt(px1*px1+py1*py1)
        rix, riy = np.arcsinh(pz1/iw1), np.arcsin(py1/iw1)
        rot_x.append(rix)
        rot_y.append(riy)        
    # now reflect if leftSum > rightSum
    leftSum = 0
    rightSum = 0
    for ix, iy, iw in zip(x, y, w):
        if ix > 0: 
            rightSum += iw
        elif ix < 0:
            leftSum += iw
    if leftSum > rightSum:
        ref_x = [-1.*rix for rix in rot_x]
        ref_y = rot_y
    else:
        ref_x = rot_x
        ref_y = rot_y
    
    return np.array(ref_x), np.array(ref_y)

def delta_phi(phi1, phi2):
  PI = 3.14159265359
  x = phi1 - phi2
  while x >=  PI:
      x -= ( 2*PI )
  while x <  -PI:
      x += ( 2*PI )
  return x

def _write_carray(a, h5file, name, group_path='/', **kwargs):
    h5file.create_carray(group_path, name, obj=a, filters=filters, createparents=True, **kwargs)


def _transform(dataframe, max_particles=150, start=0, stop=-1):
    from collections import OrderedDict
    v = OrderedDict()
    
    df = dataframe.iloc[start:stop]
    def _col_list(prefix):
        return ['%s_%d'%(prefix,i) for i in range(max_particles)]
    E = df[_col_list('E')].as_matrix()
    PX = df[_col_list('PX')].as_matrix()
    PY = df[_col_list('PY')].as_matrix()
    PZ = df[_col_list('PZ')].as_matrix()
    
    # -> PT, eta, phi
    PT = np.sqrt(PX**2 + PY**2)
    Eta = 0.5*np.log((E+PZ)/(E-PZ))
    Phi = np.arctan2(PY, PX)
    maxPtIndex = np.argmax(PT,axis=1)
    # -> Jet
    Jet_E = np.sum(E, axis=1)
    Jet_PX = np.sum(PX, axis=1)
    Jet_PY = np.sum(PY, axis=1)
    Jet_PZ = np.sum(PZ, axis=1)
    _jet_PT2 = Jet_PX**2 + Jet_PY**2
    Jet_PT = np.sqrt(_jet_PT2)
    _jet_P2 = _jet_PT2 + Jet_PZ**2
    _jet_cosTheta = Jet_PZ/np.sqrt(_jet_P2)
    Jet_Eta = -0.5*np.log( (1.0-_jet_cosTheta)/(1.0+_jet_cosTheta) )
    Jet_Phi = np.arctan2(Jet_PY, Jet_PX)
    Jet_M = np.sqrt(Jet_E**2 - _jet_P2)

    # transformed
    _label = df['is_signal_new'].as_matrix()
    v['label'] = np.stack((_label, 1-_label), axis=-1)
    v['train_val_test'] = df['ttv'].as_matrix()
    
    v['jet_px'] = Jet_PX
    v['jet_py'] = Jet_PY
    v['jet_pz'] = Jet_PZ
    v['jet_energy'] = Jet_E
    v['jet_pt'] = Jet_PT
    v['jet_eta'] = Jet_Eta
    v['jet_phi'] = Jet_Phi
    v['jet_mass'] = Jet_M

    v['part_px'] = PX
    v['part_py'] = PY
    v['part_pz'] = PZ
    v['part_energy'] = E
    v['part_pt'] = PT
    v['part_eta'] = Eta
    v['part_phi'] = Phi

    v['part_pt_log'] = np.log(PT)
    v['part_ptrel'] = PT/Jet_PT[:,None]
    v['part_ptrel_log'] = np.log(v['part_ptrel'])

    v['part_energy_log'] = np.log(E)
    v['part_erel'] = E/Jet_E[:,None]
    v['part_erel_log'] = np.log(v['part_erel'])

    _jet_etasign = np.sign(Jet_Eta)
    _jet_etasign[_jet_etasign==0] = 1
    v['part_etarel'] = (Eta - Jet_Eta[:,None]) * _jet_etasign[:,None]
    v['part_etarel_noreflect'] = (Eta - Jet_Eta[:,None])

    maxPtEta = Eta[:,0]
    maxPtPhi = Phi[:,0]
    dim = _label.shape[0]
    maxPtEta = np.repeat(np.expand_dims(maxPtEta,dim),max_particles,axis=1)
    maxPtPhi = np.repeat(np.expand_dims(maxPtPhi,dim),max_particles,axis=1)
    x = Eta - maxPtEta
    delta_phi_func = np.vectorize(delta_phi)
    y = delta_phi_func(np.array(Phi),maxPtPhi)
    w = PT
    rotate_and_reflect_func = np.vectorize(rotate_and_reflect_element)
    x_rot = []
    y_rot = []
    for ix,iy,iw in zip(x,y,w):
        x_tmp,y_tmp = rotate_and_reflect_element(ix,iy,iw)
        x_rot.append(np.nan_to_num(x_tmp))
        y_rot.append(np.nan_to_num(y_tmp))
    v['part_eta_rot'] = np.asarray(x_rot)
    v['part_phi_rot'] = np.asarray(y_rot)
    #x_vec, y_vec = rotate_and_reflect_func(x, y, w)
    _dphi = Phi - Jet_Phi[:,None]
    _pos = (np.abs(_dphi)> np.pi)
    _n = np.round(_dphi/(2*np.pi))
    _dphi[_pos] -= _n[_pos]*(2*np.pi)
    v['part_phirel'] = _dphi
    v['part_deltaR'] = np.sqrt(v['part_etarel']**2 + v['part_phirel']**2)
    v['part_costheta'] = np.cos(2.*np.arctan(np.exp(-v['part_eta'])))
    v['part_costhetarel'] = np.cos(2.*np.arctan(np.exp(-v['part_eta_rot'])))

    # fix nan/inf
    for k in v:
        if k.startswith('part_'):
            v[k][E==0]=0
    
    def _make_image(var_img, rec,isrot ,n_pixels = 64, img_ranges = [[-0.8, 0.8], [-0.8, 0.8]]):
        wgt = rec[var_img]
        if isrot:
           x = rec['part_eta_rot']
           y = rec['part_phi_rot']
        else:
           x = rec['part_etarel']
           y = rec['part_phirel']
        img = np.zeros(shape=(len(wgt), n_pixels, n_pixels))
        for i in range(len(wgt)):
            hist2d, xedges, yedges = np.histogram2d(x[i], y[i], bins=[n_pixels, n_pixels], range=img_ranges, weights=wgt[i])
            img[i] = hist2d
        return img

    v['img_pt'] = _make_image('part_ptrel', v, False,n_pixels = 224 ,img_ranges = [[-1.2, 1.2], [-1.2, 1.2]])
    v['img_pt_rot'] = _make_image('part_ptrel', v, True,n_pixels = 224, img_ranges = [[-1.2, 1.2], [-1.2, 1.2]])
    v['img_energy'] = _make_image('part_erel', v,False, n_pixels = 224, img_ranges = [[-1.2, 1.2], [-1.2, 1.2]])
    return v

def convert(source, destdir, basename, step=10000, limit=None):
    df = pd.read_hdf(source, key='table')
    logging.info('Total events:', str(df.shape[0]))
    idx=-1
    while True:
        idx+=1
        start=idx*step
        if start>=df.shape[0]: break
        if limit is not None and start>=limit: break
        if not os.path.exists(destdir):
            os.makedirs(destdir)
        output = os.path.join(destdir, '%s_%d.h5'%(basename, idx))
        logging.info(output)
        if os.path.exists(output):
            logging.warning('... file already exist: continue ...')
            continue
        with tables.open_file(output, mode='w') as h5file:
            v=_transform(df, start=start, stop=start+step)
            for k in v.keys():
                if k=='label':
                    _write_carray(v[k], h5file, name=k, title='isTop,isQCD')
                else:
                    _write_carray(v[k], h5file, name=k)
# conver training file
#convert('/data/hqu/ntuples/GMT/v0_2018_03_27/orig/train.h5', destdir='/data/hqu/ntuples/GMT/v0_2018_03_27/converted', basename='train_file')
# conver validation file
#convert('/data/hqu/ntuples/GMT/v0_2018_03_27/orig/val.h5', destdir='/data/hqu/ntuples/GMT/v0_2018_03_27/converted', basename='val_file')
# conver testing file
convert('/bigdata/shared/JetImages/v0/test.h5', destdir='/bigdata/shared/JetImages/converted/rotation_224_150p_v1', basename='test_file')
#convert('/bigdata/shared/JetImages/v0/train.h5', destdir='/bigdata/shared/JetImages/converted/rotation_224_150p_v1', basename='train_file')
#convert('/bigdata/shared/JetImages/v0/val.h5', destdir='/bigdata/shared/JetImages/converted/rotation_224_150p_v1', basename='val_file')
