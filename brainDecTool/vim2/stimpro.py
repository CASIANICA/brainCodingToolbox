# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy.misc import imsave
import cv2
from joblib import Parallel, delayed

import util as vutil
from brainDecTool.util import configParser
from brainDecTool.timeseries import hrf
from brainDecTool.math import down_sample


def mat2png(stimulus, prefix_name):
    """Comvert stimulus from mat to png format."""
    for i in range(stimulus.shape[0]):
        x = stimulus[i, :]
        new_x = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.uint8)
        new_x[..., 0] = x[0, ...]
        new_x[..., 1] = x[1, ...]
        new_x[..., 2] = x[2, ...]
        new_x = np.transpose(new_x, (1, 0, 2))
        file_name = prefix_name + '_' + str(i+1) + '.png'
        imsave(file_name, new_x)

def get_optical_flow(stimulus, prefix_name, out_dir):
    """Calculate dense optical flow from stimuli sequence."""
    img_w, img_h = stimulus.shape[2], stimulus.shape[3]
    mag = np.zeros((img_h, img_w, stimulus.shape[0]), dtype=np.float32)
    ang = np.zeros((img_h, img_w, stimulus.shape[0]), dtype=np.float32)
    for i in range(1, stimulus.shape[0]):
        if i==1:
            pim = np.zeros((img_w, img_h, 3), dtype=np.uint8)
            pim[..., 0] = stimulus[0, 0, ...]
            pim[..., 1] = stimulus[0, 1, ...]
            pim[..., 2] = stimulus[0, 2, ...]
            pim = np.transpose(pim, (1, 0, 2))
            prvs = cv2.cvtColor(pim, cv2.COLOR_BGR2GRAY)
        nim = np.zeros((img_w, img_h, 3), dtype=np.uint8)
        nim[..., 0] = stimulus[i, 0, ...]
        nim[..., 1] = stimulus[i, 1, ...]
        nim[..., 2] = stimulus[i, 2, ...]
        nim = np.transpose(nim, (1, 0, 2))
        next = cv2.cvtColor(nim, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15,
                                            3, 5, 1.2, 0)
        mtmp, atmp = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag[..., i] = mtmp
        ang[..., i] = atmp
        prvs = next
    np.save(os.path.join(out_dir, prefix_name+'_opticalflow_mag.npy'), mag)
    np.save(os.path.join(out_dir, prefix_name+'_opticalflow_ang.npy'), ang)

def cnnfeat_tr_pro(feat_dir, out_dir, dataset, layer, ds_fact=None):
    """Get TRs from CNN actiavtion datasets.
    
    Input
    -----
    feat_dir : absolute path of feature directory
    out_dir : output directory
    dataset : train or val
    layer : index of CNN layers
    ds_fact : spatial down-sample factor

    """
    # layer size
    layer_size = {1: [96, 55, 55],
                  2: [256, 27, 27],
                  3: [384, 13, 13],
                  4: [384, 13, 13],
                  5: [256, 13, 13]}
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
    else:
        feat_ts = np.load(os.path.join(feat_dir, 'stimulus_'+dataset,
                                       prefix_name+'.npy'),
                          mmap_mode='r')
        feat_ptr.append(feat_ts)
        ts_shape = feat_ts.shape

    print 'Original data shape : ', ts_shape

    # movie fps
    fps = 15
    
    # calculate down-sampled data size
    s = layer_size[layer]
    if ds_fact:
        ds_mark = '_ds%s' %(ds_fact)
        out_s = (s[0], int(np.ceil(s[1]*1.0/ds_fact)),
                 int(np.ceil(s[2]*1.0/ds_fact)), ts_shape[0]/fps)
    else:
        ds_mark = ''
        out_s = (s[0], s[1], s[2], ts_shape[0]/fps)
    print 'Down-sampled data shape : ', out_s
 
    # data array for storing time series after convolution and down-sampling
    # to save memory, a memmap is used temporally
    out_file_name = 'feat%s_%s_trs%s.npy'%(layer, dataset, ds_mark)
    out_file = os.path.join(out_dir, out_file_name)
    print 'Save TR data into file ', out_file
    feat = np.memmap(out_file, dtype='float64', mode='w+', shape=out_s)

    # convolution and down-sampling in a parallel approach
    Parallel(n_jobs=10)(delayed(stim_pro)(feat_ptr, feat, s, fps, ds_fact, i)
                        for i in range(ts_shape[1]/(s[1]*s[2])))

    # save memmap object as a numpy.array
    narray = np.array(feat)
    np.save(out_file, narray)

def stim_pro(feat_ptr, output, orig_size, fps, fact, i):
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
    bsize = orig_size[1]*orig_size[2]
    for p in range(len(feat_ptr)):
        if not p:
            ts = feat_ptr[p][:, i*bsize:(i+1)*bsize]
        else:
            ts = np.concatenate([ts, feat_ptr[p][:, i*bsize:(i+1)*bsize]],
                                axis=0)
    ts = ts.T
    #print ts.shape
    # log-transform
    ts = np.log(ts+1)
    # convolved with HRF
    convolved = np.apply_along_axis(np.convolve, 1, ts, hrf_signal)
    # remove time points after the end of the scanning run
    n_to_remove = len(hrf_times) - 1
    convolved = convolved[:, :-n_to_remove]
    # temporal down-sample
    vol_times = np.arange(0, ts.shape[1], fps)
    dconvolved = convolved[:, vol_times]
    # reshape to 3D
    dconvolved3d = dconvolved.reshape(orig_size[1], orig_size[2],
                                      len(vol_times))
    # get start index
    idx = i*bsize
    channel_idx, row, col = vutil.node2feature(idx, orig_size)

    # spatial down-sample
    if fact:
        dconvolved3d = down_sample(dconvolved3d, (fact, fact, 1))
    output[channel_idx, ...] = dconvolved3d

def feat_tr_pro(in_file, out_dir, ds_fact=None):
    """Get TRs from input 3D dataset (the third dim is time).
    
    Input
    -----
    in_file : absolute path of input file
    out_dir : output directory
    ds_fact : spatial down-sample factor

    """
    # load stimulus time courses
    feat_ts = np.load(in_file, mmap_mode='r')
    ts_shape = feat_ts.shape
    print 'Original data shape : ', ts_shape

    # scanning parameter
    TR = 1
    # movie fps
    fps = 15
    time_unit = 1.0 / fps

    # HRF config
    hrf_times = np.arange(0, 35, time_unit)
    hrf_signal = hrf.biGammaHRF(hrf_times)

    # reshape to 2D for convenience
    feat_ts = feat_ts.reshape(-1, ts_shape[2])
    # log-transform
    feat_ts = np.log(feat_ts+1)
    # convolved with HRF
    convolved = np.apply_along_axis(np.convolve, 1, feat_ts, hrf_signal)
    # remove time points after the end of the scanning run
    n_to_remove = len(hrf_times) - 1
    convolved = convolved[:, :-n_to_remove]
    # temporal down-sample
    vol_times = np.arange(0, feat_ts.shape[1], fps)
    dconvolved = convolved[:, vol_times]
    # reshape to 3D
    dconvolved3d = dconvolved.reshape(ts_shape[0], ts_shape[1], len(vol_times))

    # spatial down-sample
    if ds_fact:
        ds_mark = '_ds%s' %(ds_fact)
        dconvolved3d = down_sample(dconvolved3d, (ds_fact, ds_fact, 1))
    else:
        ds_mark = ''
    print 'Output data shape : ', dconvolved3d.shape
    
    # save TRs
    fname = os.path.basename(in_file)
    fname = fname.split('.')
    out_file_name = '.'.join([fname[0]+'_trs%s'%(ds_mark), fname[1]])
    out_file = os.path.join(out_dir, out_file_name)
    print 'Save TR data into file ', out_file
    np.save(out_file, dconvolved3d)


if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    data_dir = cf.get('base', 'path')
    cnn_dir = os.path.join(data_dir, 'stimulus')
    feat_dir = os.path.join(data_dir, 'sfeatures')

    #-- load original stimulus data
    #tf = tables.open_file(os.path.join(data_dir, 'Stimuli.mat'))
    ##tf.listNodes
    #stimulus = tf.get_node('/st')[:]
    #tf.close()
    
    #-- convert mat to png
    #mat2png(stimulus, 'train')

    #-- CNN activation pre-processing
    #cnnfeat_tr_pro(cnn_dir, feat_dir, dataset='train', layer=1, ds_fact=None)
    
    #-- calculate dense optical flow
    #get_optical_flow(stimulus, 'train', feat_dir)
    optical_file = os.path.join(feat_dir, 'train_opticalflow_mag.npy')
    feat_tr_pro(optical_file, feat_dir)
