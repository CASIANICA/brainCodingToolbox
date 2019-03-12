# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy.misc import imsave
import cv2
from joblib import Parallel, delayed
from skimage.color import rgb2gray
from skimage.measure import compare_ssim

from braincode.util import configParser
from braincode.timeseries import hrf
from braincode.math import down_sample, img_resize
from braincode.vim2 import util as vutil


def img_recon(orig_img):
    """Sugar function for image restore."""
    img_shape = orig_img.shape
    img = np.zeros((img_shape[1], img_shape[2], 3), dtype=np.uint8)
    img[..., 0] = orig_img[0, ...]
    img[..., 1] = orig_img[1, ...]
    img[..., 2] = orig_img[2, ...]
    return np.transpose(img, (1, 0, 2))

def mat2png(stimulus, prefix_name):
    """Comvert stimulus from mat to png format."""
    for i in range(stimulus.shape[0]):
        x = img_recon(stimulus[i, :])
        file_name = prefix_name + '_' + str(i+1) + '.png'
        imsave(file_name, x)

def get_optical_flow(stimulus, prefix_name, out_dir):
    """Calculate dense optical flow from stimuli sequence."""
    img_w, img_h = stimulus.shape[2], stimulus.shape[3]
    mag = np.zeros((img_h, img_w, stimulus.shape[0]), dtype=np.float32)
    ang = np.zeros((img_h, img_w, stimulus.shape[0]), dtype=np.float32)
    for i in range(1, stimulus.shape[0]):
        if i==1:
            pim = img_recon(stimulus[0, ...])
            prvs = cv2.cvtColor(pim, cv2.COLOR_BGR2GRAY)
        nim = img_recon(stimulus[i, ...])
        next = cv2.cvtColor(nim, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, 0.5, 3, 15,
                                            3, 5, 1.2, 0)
        mtmp, atmp = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag[..., i] = mtmp
        ang[..., i] = atmp
        prvs = next
    np.save(os.path.join(out_dir, prefix_name+'_opticalflow_mag.npy'), mag)
    np.save(os.path.join(out_dir, prefix_name+'_opticalflow_ang.npy'), ang)

def cnnfeat_tr_pro(feat_dir, out_dir, dataset, layer, ds_fact=None,
                   salience_modulated=False):
    """Get TRs from CNN actiavtion datasets.
    
    Input
    -----
    feat_dir : absolute path of feature directory
    out_dir : output directory
    dataset : train or val
    layer : index of CNN layers
    ds_fact : spatial down-sample factor
    salience_modulated : Boolean value

    """
    # layer size
    layer_size = {'conv1': [96, 55, 55],
                  'norm1': [96, 27, 27],
                  'conv2': [256, 27, 27],
                  'norm2': [256, 13, 13],
                  'conv3': [384, 13, 13],
                  'conv4': [384, 13, 13],
                  'conv5': [256, 13, 13],
                  'pool5': [256, 6, 6],
                  'fc6': [4096, 1, 1],
                  'fc7': [4096, 1, 1],
                  'fc8': [1000, 1, 1]}
    
    # load stimulus time courses
    prefix_name = '%s_sti_%s' % (layer, dataset)
    feat_ptr = []
    if dataset=='train':
        time_count = 0
        for i in range(10):
            tmp = np.load(os.path.join(feat_dir, dataset,
                                       prefix_name+'_'+str(i)+'.npy'),
                          mmap_mode='r')
            time_count += tmp.shape[0]
            feat_ptr.append(tmp)
        ts_shape = (time_count, feat_ptr[0].shape[1])
    else:
        feat_ts = np.load(os.path.join(feat_dir, dataset, prefix_name+'.npy'),
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
 
    # salience config
    if salience_modulated:
        sal_mark = '_salmod'
        salience_name = 'salience_%s_%s_%s.npy'%(dataset, s[1], s[2])
        salience_file = os.path.join(out_dir, salience_name)
        sal_ts = np.load(salience_file, mmap_mode='r')
        sal_ts = sal_ts.reshape(-1, sal_ts.shape[-1])
    else:
        sal_ts = None
        sal_mark = ''

    # data array for storing time series after convolution and down-sampling
    # to save memory, a memmap is used temporally
    out_file_name = '%s_%s_trs%s%s.npy'%(layer, dataset, ds_mark, sal_mark)
    out_file = os.path.join(out_dir, out_file_name)
    print 'Save TR data into file ', out_file
    feat = np.memmap(out_file, dtype='float64', mode='w+', shape=out_s)

    # convolution and down-sampling in a parallel approach
    print ts_shape[1]/(s[1]*s[2])
    Parallel(n_jobs=8)(delayed(stim_pro)(feat_ptr, feat, s, fps, ds_fact,
                                         sal_ts, i, using_hrf=True)
                          for i in range(ts_shape[1]/(s[1]*s[2])))

    # save memmap object as a numpy.array
    narray = np.array(feat)
    np.save(out_file, narray)

def stim_pro(feat_ptr, output, orig_size, fps, fact, sal_ts, i, using_hrf=True):
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
    if sal_ts:
        ts = ts * sal_ts
    # log-transform
    ts = np.log(ts+1)
    if using_hrf:
        # convolved with HRF
        convolved = np.apply_along_axis(np.convolve, 1, ts, hrf_signal)
        # remove time points after the end of the scanning run
        n_to_remove = len(hrf_times) - 1
        convolved = convolved[:, :-n_to_remove]
        # temporal down-sample
        vol_times = np.arange(0, ts.shape[1], fps)
        ndts = convolved[:, vol_times]
    else:
        # temporal down-sample
        dts = down_sample(ts, (1, fps))
        # shift time series
        ndts = np.zeros_like(dts)
        delay_time = 4
        ndts[:, delay_time:] = dts[:, :(-1*delay_time)]

    # reshape to 3D
    ndts = ndts.reshape(orig_size[1], orig_size[2], ts.shape[1]/fps)
    # get start index
    idx = i*bsize
    channel_idx, row, col = vutil.node2feature(idx, orig_size)

    # spatial down-sample
    if fact:
        ndts = down_sample(ndts, (fact, fact, 1))
    output[channel_idx, ...] = ndts

def feat_tr_pro(in_file, out_dir, out_dim=None, using_hrf=True):
    """Get TRs from input 3D dataset (the third dim is time).
    
    Input
    -----
    in_file : absolute path of input file
    out_dir : output directory
    out_dim : spatial resolution of output, a tuple of (row, col)

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
    if using_hrf:
        # convolved with HRF
        convolved = np.apply_along_axis(np.convolve, 1, feat_ts, hrf_signal)
        # remove time points after the end of the scanning run
        n_to_remove = len(hrf_times) - 1
        convolved = convolved[:, :-n_to_remove]
        # temporal down-sample
        vol_times = np.arange(0, feat_ts.shape[1], fps)
        dconvolved = convolved[:, vol_times]
    else:
        # temporal down-sample
        dts = down_sample(feat_ts, (1, fps))
        # shift time series
        dconvolved = np.zeros_like(dts)
        delay_time = 4
        dconvolved[:, delay_time:] = dts[:, :(-1*delay_time)]

    # reshape to 3D
    dconvolved3d = dconvolved.reshape(ts_shape[0], ts_shape[1], -1)

    # spatial down-sample
    if out_dim:
        ds_mark = '_%s_%s' %(out_dim[0], out_dim[1])
        dconvolved3d = img_resize(dconvolved3d, out_dim)
        #im_min, im_max = dconvolved3d.min(), dconvolved3d.max()
        #im_std = (dconvolved3d - im_min) / (im_max - im_min)
        #resized_im = resize(im_std, out_dim, order=1)
        #dconvolved3d = resized_im * (im_max - im_min) + im_min
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

def get_stim_seq(stimulus, output_filename):
    """Get stimulus sequence by compute structural similarity between
    adjacent frames.
    """
    fps = 15 
    stim_len = stimulus.shape[0]
    stim_seq = np.zeros((stim_len,))

    img_w, img_h = stimulus.shape[2], stimulus.shape[3]
    for i in range(1, stim_len):
        pim = img_recon(stimulus[i-1, ...])
        nim = img_recon(stimulus[i, ...]) 
        pim = rgb2gray(pim)
        nim = rgb2gray(nim)
        stim_seq[i] = compare_ssim(pim, nim, multichannel=False)
    
    # convolved with HRF
    time_unit = 1.0 / fps
    # HRF config
    hrf_times = np.arange(0, 35, time_unit)
    hrf_signal = hrf.biGammaHRF(hrf_times)
    convolved_seq = np.convolve(stim_seq, hrf_signal)
    # remove time points after the end of the scanning run
    n_to_remove = len(hrf_times) - 1
    convolved_seq = convolved_seq[:-n_to_remove]
    # temporal down-sample
    vol_times = np.arange(0, stim_len, fps)
    dseq = convolved_seq[vol_times]
    np.save(output_filename, dseq)

def gaussian_kernel_feats(feat_dir):
    """Get CNN features modulated by Gaussian kernels."""
    # load CNN features
    train_feat_file = os.path.join(feat_dir, 'conv1_train_trs.npy')
    feat_ts = np.load(train_feat_file, mmap_mode='r')
    #val_feat_file = os.path.join(feat_dir, 'conv1_val_trs.npy')
    #feat_ts = np.load(val_feat_file, mmap_mode='r')
    # load Gaussian kernels
    kernel_file = os.path.join(feat_dir, 'gaussian_prfs.npy')
    kernels = np.load(kernel_file)
    # data reshape
    feat_ts = feat_ts.reshape(96, 3025, 7200)
    feat_ts = np.transpose(feat_ts, (0, 2, 1))
    #feat_ts = feat_ts.reshape(96, 3025, 540)
    #feat_ts = np.transpose(feat_ts, (0, 2, 1))
    kernels = kernels.reshape(3025, 30250)
    # calculate gaussian modulated features
    outdir = os.path.join(feat_dir, 'gaussian_kernels')
    os.system('mkdir %s'%(outdir))
    for i in range(30250/550):
        print 'Iter %s/%s'%(i+1, 30250/550)
        ndata = np.zeros((96, 7200, 550))
        ktmp = kernels[:, i*550:(i+1)*550]
        for j in range(96):
            print 'Channel %s'%(j+1)
            ndata[j, ...] = np.dot(feat_ts[j, ...], ktmp)
        outfile = os.path.join(outdir, 'gaussian_conv1_train_trs_%s.npy'%(i))
        np.save(outfile, ndata)


if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    data_dir = cf.get('base', 'path')
    stim_dir = os.path.join(data_dir, 'stimulus', 'vim2')
    feat_dir = os.path.join(data_dir, 'sfeatures', 'vim2')

    #-- load original stimulus data
    #tf = tables.open_file(os.path.join(stim_dir, 'Stimuli.mat'))
    #data_type = 'train'
    #data_dic = {'train': '/st', 'val': '/sv'}
    #stimulus = tf.get_node(data_dic[data_type])[:]
    #tf.close()

    #-- get stimulus sequence
    #get_stim_seq(stimulus, 'conv_gray_stim_train_design.npy')

    #-- convert mat to png
    #mat2png(stimulus, 'train')

    #-- CNN activation pre-processing
    cnnfeat_tr_pro(stim_dir, feat_dir, dataset='train', layer='fc6',
                   ds_fact=None, salience_modulated=False)
    
    #-- calculate dense optical flow
    #get_optical_flow(stimulus, 'train', feat_dir)
    #optical_file = os.path.join(feat_dir, 'train_opticalflow_mag.npy')
    #feat_tr_pro(optical_file, feat_dir, out_dim=None, using_hrf=False)

    #-- salience processing
    #sal_file = os.path.join(feat_dir, 'salience_val_55_55.npy')
    #feat_tr_pro(sal_file, feat_dir, out_dim=None, using_hrf=True)

    #gaussian_kernel_feats(feat_dir)
    
