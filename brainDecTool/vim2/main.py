# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy import ndimage
from scipy.misc import imsave

from joblib import Parallel, delayed

from brainDecTool.util import configParser
from brainDecTool.pipeline import retinotopy
from brainDecTool.pipeline.base import cross_modal_corr
from brainDecTool.math import down_sample
from brainDecTool.timeseries import hrf
import util as vutil


def feat_down_sample(feat_dir, dataset, layer, out_dir, fact):
    """Spatial down-sampling."""
    # layer size
    layer_size = {1: [96, 55, 55],
                  2: [256, 27, 27],
                  3: [384, 13, 13],
                  4: [384, 13, 13],
                  5: [256, 13, 13]}
    # load stimulus time courses
    prefix_name = 'feat%s_sti_%s' % (layer, dataset)
    feat_ptr = []
    start_idx = []
    if dataset=='train':
        time_count = 0
        for i in range(12):
            tmp = np.load(os.path.join(feat_dir, 'stimulus_'+dataset,
                                       prefix_name+'_'+str(i+1)+'.npy'),
                          mmap_mode='r')
            start_idx.append(time_count)
            time_count += tmp.shape[0]
            feat_ptr.append(tmp)
        ts_shape = (time_count, feat_ptr[0].shape[1])
    else:
        feat_ts = np.load(os.path.join(feat_dir, 'stimulus_'+datatset,
                                       prefix_name+'.npy'),
                          mmap_mode='r')
        feat_ptr.append(feat_ts)
        start_idx.append(0)
        ts_shape = feat_ts.shape
    print 'Original data shape : ', ts_shape
    
    # calculate down-sampled data size
    s = layer_size[layer]
    new_shape = (ts_shape[0], int(s[0]*(np.ceil(s[1]*1.0/fact)**2)))
    print 'Down-sampled data shape : ', new_shape

    # open a new file for data storage
    out_file = os.path.join(out_dir,
                'feat%s_%s_ds%s.npy'%(layer, dataset, fact))
    print 'Save down-sampled data into file ', out_file
    feat = np.memmap(out_file, dtype='float64', mode='w+', shape=new_shape)

    # down-sampling processing in a parallel approach
    Parallel(n_jobs=12)(delayed(down_sample_pro)(feat_ptr[i], s, fact,
                                start_idx[i], feat)
                                for i in range(len(feat_ptr)))

def down_sample_pro(orig_feat, orig_size, fact, start_idx, output):
    """Sugar function for parallel computing."""
    data_shape = orig_feat.shape
    for i in range(data_shape[0]):
        tmp = orig_feat[i, :].reshape(orig_size)
        dtmp = down_sample(tmp, (1, fact, fact))
        output[start_idx+i, :] = dtmp.flatten()

def feat_tr_pro(feat_dir, out_dir,  data_file=None, data_size=None,
                dataset=None, layer=None, log_trans=True):
    """Get TRs from CNN actiavtion datasets.
    
    Input
    -----
    feat_dir : absolute path of feature directory
    dataset : train or val
    layer : index of CNN layers
    out_dir : output directory

    """
    # load stimulus time courses
    prefix_name = 'feat%s_sti_%s' % (layer, dataset)
    feat_ptr = []
    if dataset=='train':
        time_count = 0
        for i in range(12):
            tmp = np.load(os.path.join(feat_dir, 'stimulus_'+dataset,
                                       prefix_name+'_'+str(i+1)+'.npy'),
                          mmap_mode='r')
            time_count += tmp.shape[0]
            feat_ptr.append(tmp)
        ts_shape = (time_count, feat_ptr[0].shape[1])
    elif dataset=='val':
        feat_ts = np.load(os.path.join(feat_dir, 'stimulus_'+datatset,
                                       prefix_name+'.npy'),
                          mmap_mode='r')
        feat_ptr.append(feat_ts)
        ts_shape = feat_ts.shape
    else:
        feat_ts = np.memmap(os.path.join(feat_dir, data_file),
                            dtype='float64', mode='r',
                            shape=data_size)
        feat_ptr.append(feat_ts)
        ts_shape = feat_ts.shape

    print 'Original data shape : ', ts_shape

    # movie fps
    fps = 15
    
    # data array for storing time series after convolution and down-sampling
    # to save memory, a memmap is used
    if log_trans:
        log_mark = '_log'
    else:
        log_mark = ''
    out_file_name = 'feat%s_%s_trs%s.npy'%(layer, dataset, log_mark)
    out_file = os.path.join(out_dir, out_file_name)
    print 'Save TR data into file ', out_file
    feat = np.memmap(out_file, dtype='float64', mode='w+',
                     shape=(ts_shape[1], ts_shape[0]/fps))

    # batch_size config
    bsize = 96
    # convolution and down-sampling in a parallel approach
    Parallel(n_jobs=10)(delayed(stim_pro)(feat_ptr, feat, bsize, fps, log_trans,
                            i) for i in range(ts_shape[1]/bsize))

def stim_pro(feat_ptr, output, bsize, fps, log_trans, i):
    """Sugar function for parallel computing."""
    print i
    # scanning parameter
    TR = 1
    # movie fps
    #fps = 15
    time_unit = 1.0 / fps

    # HRF config
    hrf_times = np.arange(0, 35, time_unit)
    hrf_signal = hrf.biGammaHRF(hrf_times)

    # procssing
    for p in range(len(feat_ptr)):
        if not p:
            ts = feat_ptr[p][:, i*bsize:(i+1)*bsize]
        else:
            ts = np.concatenate([ts, feat_ptr[p][:, i*bsize:(i+1)*bsize]],
                                axis=0)
    ts = ts.T
    #print ts.shape
    # log-transform
    if log_trans:
        ts = np.log(ts+1)
    # convolved with HRF
    convolved = np.apply_along_axis(np.convolve, 1, ts, hrf_signal)
    # remove time points after the end of the scanning run
    n_to_remove = len(hrf_times) - 1
    convolved = convolved[:, :-n_to_remove]
    # down-sampling
    vol_times = np.arange(0, ts.shape[1], fps)
    output[i*bsize:(i+1)*bsize, :] = convolved[:, vol_times]

def roi2nifti(fmri_table):
    """Save ROI as a nifti file."""
    l_v1_roi = fmri_table.get_node('/roi/v1lh')[:]
    r_v1_roi = fmri_table.get_node('/roi/v1rh')[:]
    l_v2_roi = fmri_table.get_node('/roi/v2lh')[:]
    r_v2_roi = fmri_table.get_node('/roi/v2rh')[:]
    l_v3_roi = fmri_table.get_node('/roi/v3lh')[:]
    r_v3_roi = fmri_table.get_node('/roi/v3rh')[:]
    l_v3a_roi = fmri_table.get_node('/roi/v3alh')[:]
    r_v3a_roi = fmri_table.get_node('/roi/v3arh')[:]
    # merge ROIs in both hemisphere
    roi_mask = l_v1_roi + r_v1_roi*2 + l_v2_roi*3 + r_v2_roi*4 + \
               l_v3_roi*5 + r_v3_roi*6 + l_v3a_roi*7 + r_v3a_roi*8
    vutil.save2nifti(roi_mask, 'S1_roi_mask.nii.gz')

def gen_mean_vol(fmri_table):
    """Make a mean response map as a reference volume."""
    data = fmri_table.get_node('/rt')[:]
    # replace nan to zero
    data = np.nan_to_num(data)
    mean_data = np.mean(data, axis=1)
    vol = np.zeros((18, 64, 64))
    
    for i in range(data.shape[0]):
        c = vutil.idx2coord(i)
        vol[c[0], c[1], c[2]] = mean_data[i]
    
    vutil.save2nifti(vol, 'S1_mean_rt.nii.gz')

def retinotopic_mapping(corr_file):
    """Make the retinotopic mapping using activation map from CNN."""
    data_dir = os.path.dirname(corr_file)
    #fig_dir = os.path.join(data_dir, 'fig')
    #if not os.path.exists(fig_dir):
    #    os.mkdir(fig_dir, 0755)
    # load the cross-correlation matrix from file
    #corr_mtx = np.load(corr_file, mmap_mode='r')
    corr_mtx = np.memmap(corr_file, dtype='float16', mode='r',
                         shape=(73728, 290400))
    pos_mtx = np.zeros((corr_mtx.shape[0], 2))
    for i in range(corr_mtx.shape[0]):
        print 'Iter %s of %s' %(i, corr_mtx.shape[0]),
        tmp = corr_mtx[i, :]
        tmp = np.nan_to_num(np.array(tmp))
        if np.sum(tmp):
            tmp = tmp.reshape(96, 55, 55)
            mmtx = np.max(tmp, axis=0)
            print mmtx.min(), mmtx.max()
            #fig_file = os.path.join(fig_dir, 'v'+str(i)+'.png')
            #imsave(fig_file, mmtx)
            # get indices of n maximum values
            max_n = 20
            row_idx, col_idx = np.unravel_index(
                                        np.argsort(mmtx.ravel())[-1*max_n:],
                                        mmtx.shape)
            nmtx = np.zeros(mmtx.shape)
            nmtx[row_idx, col_idx] = mmtx[row_idx, col_idx]
            # center of mass
            x, y = ndimage.measurements.center_of_mass(nmtx)
            pos_mtx[i, :] = [x, y]
        else:
            pos_mtx[i, :] = [np.nan, np.nan]
            print ' '
    #receptive_field_file = os.path.join(data_dir, 'receptive_field_pos.npy')
    #np.save(receptive_field_file, pos_mtx)
    #pos_mtx = np.load(receptive_field_file)
    # eccentricity
    dist = retinotopy.coord2ecc(pos_mtx, (55, 55))
    # convert distance into degree
    # 0-4 degree -> d < 5.5
    # 4-8 degree -> d < 11
    # 8-12 degree -> d < 16.5
    # 12-16 degree -> d < 22
    # else > 16 degree
    ecc = np.zeros(dist.shape)
    for i in range(len(dist)):
        if dist[i] < 5.445:
            ecc[i] = 1
        elif dist[i] < 10.91:
            ecc[i] = 2
        elif dist[i] < 16.39:
            ecc[i] = 3
        elif dist[i] < 21.92:
            ecc[i] = 4
        else:
            ecc[i] = 5
    dist_vec = np.nan_to_num(ecc)
    vol = dist_vec.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max' + str(max_n) + '_ecc.nii.gz'))
    # angle
    angle_vec = retinotopy.coord2angle(pos_mtx, (55, 55))
    angle_vec = np.nan_to_num(angle_vec)
    vol = angle_vec.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max'+ str(max_n) + '_angle.nii.gz'))

def hrf_estimate(tf, feat_ts):
    """Estimate HRFs."""
    # voxel coordinates for test
    # voxels from R_V1
    # 0. (20, 36, 13) -> (20, 27, 13) -> (13, 20, 27)
    # 1. (20, 34, 13) -> (20, 29, 13) -> (13, 20, 29)
    # 2. (22, 34, 9) -> (22, 29, 9) -> (9, 22, 29)
    # voxels from L_V1
    # 3. (20, 29, 9) -> (20, 34, 9) -> (9, 20, 34)
    # 4. (16, 29, 12) -> (16, 34, 12) -> (12, 16, 34)

    voxels = [(20, 36, 13),
              (20, 34, 13),
              (22, 34, 9),
              (20, 29, 9),
              (16, 29, 12)]
    # voxel label validation
    #rv1_roi = tf.get_node('/roi/v1rh')
    #lv1_roi = tf.get_node('/roi/v1lh')
    # get time courses for each voxel
    vxl_idx = [vutil.coord2idx(coord) for coord in voxels]
    rt = tf.get_node('/rt')[:]
    vxl_data = rt[vxl_idx, :]
    vxl_data = np.nan_to_num(vxl_data)

    out = np.zeros((290400, 40, 5))
    for i in range(5):
        for j in range(feat_ts.shape[0]):
            print '%s - %s' %(i, j)
            tmp = feat_ts[j, :]
            tmp = (tmp - tmp.mean()) / tmp.std()
            out[j, :, i] = time_lag_corr(tmp, vxl_data[i, :], 40)
    np.save('hrf_test.npy', out)

if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    data_dir = cf.get('base', 'path')
    feat_dir = os.path.join(data_dir, 'stimulus')
    stim_dir = os.path.join(data_dir, 'cnn_rsp')

    #-- spatial down-sample stimulus
    #feat_down_sample(feat_dir, 'train', 1, stim_dir, 5)

    #-- CNN activation pre-processing
    #feat_tr_pro(feat_dir, stim_dir, dataset='train', layer=1, log_trans=False)
    #feat_tr_pro(stim_dir, stim_dir, data_file='feat1_train_ds5.npy',
    #            data_size=(108000, 11616), log_trans=False)
    
    #-- load fmri data
    tf = tables.open_file(os.path.join(data_dir, 'VoxelResponses_subject1.mat'))
    #tf.list_nodes
    #-- mat to nii
    #roi2nifti(tf)
    #gen_mean_vol(tf)
    #-- to be used
    #roi = tf.get_node('/roi/v1lh')[:].flatten()
    #v1lh_idx = np.nonzero(roi==1)[0]
    #v1lh_resp = data[v1lh_idx]

    #-- calculate cross-modality corrlation 
    # load fmri response from training/validation dataset
    fmri_ts = tf.get_node('/rt')[:]
    # data.shape = (73728, 540) / (73728, 7200)
    fmri_ts = np.nan_to_num(fmri_ts)
    # load convolved cnn activation data for validation dataset
    feat1_file = os.path.join(stim_dir, 'feat1_train_trs_log.npy')
    #feat1_ts = np.load(feat1_file, mmap_mode='r')
    feat1_ts = np.memmap(feat1_file, dtype='float64', mode='r',
                         #shape=(11616, 7200))
                         shape=(290400, 7200))
    # sum up all channels
    feat1_ts = feat1_ts.reshape(96, 55, 55, 7200)
    feat1_ts = feat1_ts.sum(axis=0)
    feat1_ts = feat1_ts.reshape(3025, 7200)
    # data.shape = (290400, 540)/(290400, 7200)
    retino_dir = os.path.join(data_dir, 'retinotopic')
    if not os.path.exists(retino_dir):
        os.mkdir(retino_dir, 0755)
    corr_file = os.path.join(retino_dir, 'train_fmri_feat1_ds5_corr.npy')
    cross_modal_corr(fmri_ts, feat1_ts, corr_file, block_size=96)
    
    #-- retinotopic mapping
    #retinotopic_mapping(corr_file)

    #-- close fmri data
    tf.close()

