# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy.misc import imsave
import cv2
from joblib import Parallel, delayed
import bob.ip.gabor

from brainDecTool.util import configParser
from brainDecTool.math import ipl
from brainDecTool.timeseries import hrf
from brainDecTool.math import down_sample, img_resize


def img_recon(orig_img):
    """Reorder data shape to RGB sequence."""
    img_shape = orig_img.shape
    img = np.zeros((img_shape[1], img_shape[2], 3), dtype=np.uint8)
    img[..., 0] = orig_img[0, ...]
    img[..., 1] = orig_img[1, ...]
    img[..., 2] = orig_img[2, ...]
    return np.transpose(img, (1, 0, 2))

def mat2png(stimulus, prefix_name):
    """Comvert stimulus from mat to png format."""
    x = np.transpost(stimulus, (0, 3, 2, 1))
    for i in range(x.shape[0]):
        file_name = prefix_name + '_' + str(i+1) + '.png'
        imsave(file_name, x[i, ...])

def get_gabor_features(img):
    """Get Gabor features from input image."""
    img = img.astype(np.float64)
    gwt = bob.ip.gabor.Transform()
    trafo_img = gwt(img)
    out_feat = np.zeros((img.shape[0], img.shape[1], trafo_img.shape[0]))
    for i in range(trafo_img.shape[0]):
        real_p = np.real(trafo_img[i, ...])
        imag_p = np.imag(trafo_img[i, ...])
        out_feat[..., i] = np.sqrt(np.square(real_p) + np.square(imag_p))
    return out_feat

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

def feat2bold(feat_dir, dataset, ftype):
    """Convert feature time course to expected BOLD signals.
    
    Input
    -----
    feat_dir : absolute path of feature directory
    dataset : train or val
    ftype : gabor or hue
    ds_fact : spatial down-sample factor

    """
    # load stimulus time courses
    prefix_name = '%s_%s' % (dataset, ftype)
    feat_ptr = []
    if dataset=='train':
        time_count = 0
        for i in range(15):
            tmp = np.load(os.path.join(feat_dir, prefix_name+'_'+str(i)+'.npy'),
                          mmap_mode='r')
            time_count += tmp.shape[0]
            feat_ptr.append(tmp)
        ts_shape = (time_count, feat_ptr[0].shape[1],
                    feat_ptr[0].shape[2], feat_ptr[0].shape[3])
    else:
        feat_ts = np.load(os.path.join(feat_dir, prefix_name+'.npy'),
                          mmap_mode='r')
        feat_ptr.append(feat_ts)
        ts_shape = feat_ts.shape
    print 'Original data shape : ', ts_shape

    # movie fps
    fps = 15
    
    # calculate spatial down-sampled size
    out_s = (ts_shape[0]/fps, ts_shape[1], ts_shape[2], ts_shape[3])
    print 'Down-sampled data shape : ', out_s

    # data array for storing time series after convolution and down-sampling
    # to save memory, a memmap is used temporally
    out_file_name = '%s_%s_trs.npy'%(dataset, ftype)
    out_file = os.path.join(feat_dir, out_file_name)
    print 'Save TR data into file ', out_file
    bold = np.memmap(out_file, dtype='float16', mode='w+', shape=out_s)

    # convolution and down-sampling in a parallel approach
    Parallel(n_jobs=8)(delayed(stim_pro)(feat_ptr, bold, ts_shape, fps,
                                         i, using_hrf=True)
                        for i in range(ts_shape[2]*ts_shape[3]))

    # save memmap object as a numpy.array
    narray = np.array(bold)
    np.save(out_file, narray)

def stim_pro(feat_ptr, output, orig_size, fps, i, using_hrf=True):
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
    hrf_signal = hrf_signal.astype(np.float16)

    # procssing
    channel_idx = i / orig_size[2]
    col_idx = i % orig_size[2]
    tmp_list = []
    for p in feat_ptr:
        tmp_list.append(p[..., col_idx, channel_idx])
    ts = np.concatenate(tmp_list, axis=0)
    del tmp_list
    # log-transform
    # memory saving trick
    ts += 1
    ts = np.log(ts.T)
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
    output[..., col_idx, channel_idx] = ndts.T

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
    stim_dir = os.path.join(data_dir, 'stimulus')
    feat_dir = os.path.join(data_dir, 'sfeatures')

    # load original stimulus data
    #tf = tables.open_file(os.path.join(stim_dir, 'Stimuli.mat'))
    #data_type = 'train'
    #data_dic = {'train': '/st', 'val': '/sv'}
    #stimulus = tf.get_node(data_dic[data_type])[:]
    #tf.close()
    
    # convert mat to png
    #mat2png(stimulus, data_type)
    
    # convert images from RGB to CIRLCh space
    #stimulus = np.transpose(stimulus, (0, 3, 2, 1))
    #ciesti = np.zeros_like(stimulus, dtype=np.float16)
    #for i in range(stimulus.shape[0]):
    #    print i
    #    tmp = ipl.rgb2cielab(stimulus[i, ...])
    #    ciesti[i, ...] = ipl.cielab2cielch(tmp)
    #np.save(os.path.join(stim_dir, data_type+'_stimulus_LCh.npy'), ciesti)
    
    # extract hue features
    #data_type = 'train'
    #sti_file = os.path.join(stim_dir, data_type+'_stimulus_LCh.npy')
    #sti = np.load(sti_file, mmap_mode='r')
    ## we define 8 hue basis, each corresponding pi/4 range of hue
    #if data_type=='train':
    #    parts = 15
    #    num_parts = sti.shape[0] / parts
    #    for i in range(parts):
    #        hue_feat = np.zeros((num_parts, sti.shape[1], sti.shape[2], 6),
    #                            dtype=np.float16)
    #        hue = sti[i*num_parts:(i+1)*num_parts, ..., 2].copy()
    #        hue[hue<0] += 2*np.pi
    #        for j in range(6):
    #            tmp = np.sin(hue-j*np.pi/3)
    #            tmp[tmp<0] = 0
    #            hue_feat[..., j] = np.square(tmp) 
    #        ofile = os.path.join(feat_dir, data_type+'_hue_%s.npy'%(i))
    #        np.save(ofile, hue_feat)
    #else:
    #    hue_feat = np.zeros((sti.shape[0], sti.shape[1], sti.shape[2], 6),
    #                        dtype=np.float16)
    #    hue = sti[..., 2].copy()
    #    hue[hue<0] += 2*np.pi
    #    for j in range(6):
    #        tmp = np.sin(hue-j*np.pi/3)
    #        tmp[tmp<0] = 0
    #        hue_feat[..., j] = np.square(tmp) 
    #    ofile = os.path.join(feat_dir, data_type+'_hue.npy')
    #    np.save(ofile, hue_feat)

    # extract gabor features
    #data_type = 'val'
    #sti_file = os.path.join(stim_dir, data_type+'_stimulus_LCh.npy')
    #sti = np.load(sti_file, mmap_mode='r')
    ## we define 40 gabor basis
    #if data_type=='train':
    #    parts = 15
    #    num_parts = sti.shape[0] / parts
    #    for i in range(parts):
    #        gabor_feat = np.zeros((num_parts, sti.shape[1], sti.shape[2], 40),
    #                              dtype=np.float16)
    #        L = sti[i*num_parts:(i+1)*num_parts, ..., 0].copy()
    #        for j in range(num_parts):
    #            x = L[j, ...] / 100 * 255
    #            gabor_feat[j, ...] = get_gabor_features(x)
    #        ofile = os.path.join(feat_dir, data_type+'_gabor_%s.npy'%(i))
    #        np.save(ofile, gabor_feat)
    #else:
    #    gabor_feat = np.zeros((sti.shape[0], sti.shape[1], sti.shape[2], 40),
    #                          dtype=np.float16)
    #    L = sti[..., 0].copy()
    #    for j in range(L.shape[0]):
    #        x = L[j, ...] / 100 * 255
    #        gabor_feat[j, ...] = get_gabor_features(x)
    #    ofile = os.path.join(feat_dir, data_type+'_gabor.npy')
    #    np.save(ofile, gabor_feat)

    # features to expected BOLD signal
    feat2bold(feat_dir, dataset='val', ftype='hue')
    
    #-- calculate dense optical flow
    #get_optical_flow(stimulus, 'train', feat_dir)
    #optical_file = os.path.join(feat_dir, 'train_opticalflow_mag.npy')
    #feat_tr_pro(optical_file, feat_dir, out_dim=None, using_hrf=False)

    #gaussian_kernel_feats(feat_dir)
    
