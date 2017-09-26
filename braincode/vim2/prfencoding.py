# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy.misc import imsave
from joblib import Parallel, delayed
import bob.ip.gabor

from braincode.util import configParser
from braincode.math import ipl, make_2d_gaussian, ridge
from braincode.timeseries import hrf
from braincode.math.norm import zscore
from braincode.vim2 import dataio

from sklearn.linear_model import LassoCV

def check_path(dir_path):
    """Check whether the directory does exist, if not, create it."""            
    if not os.path.exists(dir_path):
        os.mkdir(dir_path, 0755)

def mat2png(stimulus, prefix_name):
    """Comvert stimulus from mat to png format."""
    x = np.transpost(stimulus, (3, 2, 1, 0))
    for i in range(x.shape[0]):
        file_name = prefix_name + '_' + str(i+1) + '.png'
        imsave(file_name, x[..., i])

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
            time_count += tmp.shape[3]
            feat_ptr.append(tmp)
        ts_shape = (feat_ptr[0].shape[0], feat_ptr[0].shape[1],
                    feat_ptr[0].shape[2], time_count)
    else:
        feat_ts = np.load(os.path.join(feat_dir, prefix_name+'.npy'),
                          mmap_mode='r')
        feat_ptr.append(feat_ts)
        ts_shape = feat_ts.shape
    print 'Original data shape : ', ts_shape

    # movie fps
    fps = 15
    
    # calculate spatial down-sampled size
    out_s = (ts_shape[0], ts_shape[1], ts_shape[2], ts_shape[3]/fps)
    print 'Down-sampled data shape : ', out_s

    # data array for storing time series after convolution and down-sampling
    # to save memory, a memmap is used temporally
    out_file_name = '%s_%s_trs.npy'%(dataset, ftype)
    out_file = os.path.join(feat_dir, out_file_name)
    print 'Save TR data into file ', out_file
    bold = np.memmap(out_file, dtype='float16', mode='w+', shape=out_s)

    # convolution and down-sampling in a parallel approach
    Parallel(n_jobs=10)(delayed(stim_pro)(feat_ptr, bold, ts_shape, fps, i)
                        for i in range(ts_shape[0]*ts_shape[1]))

    # save memmap object as a numpy.array
    narray = np.array(bold)
    np.save(out_file, narray)

def stim_pro(feat_ptr, output, orig_size, fps, i):
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
    r_idx = i / orig_size[1]
    c_idx = i % orig_size[1]
    tmp_list = []
    for p in feat_ptr:
        tmp_list.append(p[r_idx, c_idx, ...])
    ts = np.concatenate(tmp_list, axis=1)
    # log-transform
    # memory saving trick
    ts += 1
    ts = np.log(ts)
    # convolved with HRF
    convolved = np.apply_along_axis(np.convolve, 1, ts, hrf_signal)
    # remove time points after the end of the scanning run
    n_to_remove = len(hrf_times) - 1
    convolved = convolved[:, :-n_to_remove]
    # temporal down-sample
    vol_times = np.arange(0, ts.shape[1], fps)
    output[r_idx, c_idx, ...] = convolved[:, vol_times]

def get_candidate_model(feat_dir):
    """ Get gaussian kernel based receptive field features"""
    # load feature data
    train_gabor = np.load(os.path.join(feat_dir, 'train_gabor_trs.npy'))
    train_hue = np.load(os.path.join(feat_dir, 'train_hue_trs.npy'))
    val_gabor = np.load(os.path.join(feat_dir, 'val_gabor_trs.npy'))
    val_hue = np.load(os.path.join(feat_dir, 'val_hue_trs.npy'))
    train_feat = np.concatenate([train_gabor, train_hue], axis=2)
    train_feat = train_feat.reshape(128*128, 46*7200)
    del train_gabor
    del train_hue
    val_feat = np.concatenate([val_gabor, val_hue], axis=2)
    val_feat = val_feat.reshape(128*128, 46*540)
    del val_gabor
    del val_hue
    # derived gaussian-based features
    # candidate pooling centers are spaced 0.625 degrees apart (4 pixels)
    # candiante pooling fields included 15 evenly-spaces radii between 0.16
    # degrees (1 pixel) and 7.8 degrees (50 pixels)
    out_train = os.path.join(feat_dir, 'train_candidate_model.npy')
    out_val = os.path.join(feat_dir, 'val_candidate_model.npy')
    train_model = np.memmap(out_train, dtype='float16', mode='w+',
                            shape=(32*32*15, 46, 7200))
    val_model = np.memmap(out_val, dtype='float16', mode='w+',
                          shape=(32*32*15, 46, 540))
    Parallel(n_jobs=3)(delayed(model_pro)(train_feat, val_feat, train_model,
                                          val_model, xi, yi, si)
                    for si in range(15) for xi in range(32) for yi in range(32))
    
    # save memmap object as a numpy.array
    train_array = np.array(train_model)
    np.save(out_train, train_array)
    val_array = np.array(val_model)
    np.save(out_val, val_array)

def model_pro(train_in, val_in, train_out, val_out, xi, yi, si):
    """Sugar function for generating  candidate model"""
    mi = si*32*32 + xi*32 + yi
    center_x = np.arange(0, 128, 4)
    center_y = np.arange(0, 128, 4)
    sigma = np.linspace(1, 50, 15)
    x0 = center_x[xi]
    y0 = center_y[yi]
    s = sigma[si]
    print 'Model %s : center - (%s, %s), sigma %s'%(mi, x0, y0, s)
    kernel = make_2d_gaussian(128, s, center=(x0, y0))
    kernel = kernel.flatten()
    tmp = np.zeros((331200, ), dtype=np.float16)
    for i in range(23):
        m = i*14400
        n = m + 14400
        tmp[m:n] = kernel.dot(train_in[:, m:n]).astype(np.float16)
    train_out[mi] = tmp.reshape(46, 7200)
    val_out[mi] = kernel.dot(val_in).reshape(46, 540).astype(np.float16)

if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    root_dir = cf.get('base', 'path')
    stim_dir = os.path.join(root_dir, 'stimulus')
    feat_dir = os.path.join(root_dir, 'sfeatures')
    db_dir = os.path.join(root_dir, 'subjects')

    # load original stimulus data
    #tf = tables.open_file(os.path.join(stim_dir, 'Stimuli.mat'))
    #data_type = 'train'
    #data_dic = {'train': '/st', 'val': '/sv'}
    #stimulus = tf.get_node(data_dic[data_type])[:]
    #tf.close()
    
    # convert mat to png
    #mat2png(stimulus, data_type)
    
    # convert images from RGB to CIRLCh space
    #stimulus = np.transpose(stimulus, (3, 2, 1, 0))
    #ciesti = np.zeros_like(stimulus, dtype=np.float16)
    #for i in range(stimulus.shape[3]):
    #    tmp = ipl.rgb2cielab(stimulus[..., i])
    #    ciesti[..., i] = ipl.cielab2cielch(tmp)
    #np.save(os.path.join(stim_dir, data_type+'_stimulus_LCh.npy'), ciesti)
    
    # extract hue features
    #data_type = 'val'
    #sti_file = os.path.join(stim_dir, data_type+'_stimulus_LCh.npy')
    #sti = np.load(sti_file, mmap_mode='r')
    ## we define 6 hue basis, each corresponding pi range of hue, with a pi/3
    ## difference in phrase
    #if data_type=='train':
    #    parts = 15
    #    part_size = sti.shape[3] / parts
    #    for i in range(parts):
    #        hue_feat = np.zeros((sti.shape[0], sti.shape[1], 6, part_size),
    #                            dtype=np.float16)
    #        hue = sti[..., 2, i*part_size:(i+1)*part_size].copy()
    #        hue[hue<0] += 2*np.pi
    #        for j in range(6):
    #            tmp = np.sin(hue-j*np.pi/3)
    #            tmp[tmp<0] = 0
    #            hue_feat[..., j, :] = np.square(tmp) 
    #        ofile = os.path.join(feat_dir, data_type+'_hue_%s.npy'%(i))
    #        np.save(ofile, hue_feat)
    #else:
    #    hue_feat = np.zeros((sti.shape[0], sti.shape[1], 6, sti.shape[3]),
    #                        dtype=np.float16)
    #    hue = sti[..., 2, :].copy()
    #    hue[hue<0] += 2*np.pi
    #    for j in range(6):
    #        tmp = np.sin(hue-j*np.pi/3)
    #        tmp[tmp<0] = 0
    #        hue_feat[..., j, :] = np.square(tmp) 
    #    ofile = os.path.join(feat_dir, data_type+'_hue.npy')
    #    np.save(ofile, hue_feat)

    # extract gabor features
    #data_type = 'val'
    #sti_file = os.path.join(stim_dir, data_type+'_stimulus_LCh.npy')
    #sti = np.load(sti_file, mmap_mode='r')
    ## we define 40 gabor basis
    #if data_type=='train':
    #    parts = 15
    #    part_size = sti.shape[3] / parts
    #    for i in range(parts):
    #        gabor_feat = np.zeros((sti.shape[0], sti.shape[1], 40, part_size),
    #                              dtype=np.float16)
    #        L = sti[..., 0, i*part_size:(i+1)*part_size].copy()
    #        for j in range(part_size):
    #            x = L[..., j] / 100 * 255
    #            gabor_feat[..., j] = get_gabor_features(x)
    #        ofile = os.path.join(feat_dir, data_type+'_gabor_%s.npy'%(i))
    #        np.save(ofile, gabor_feat)
    #else:
    #    gabor_feat = np.zeros((sti.shape[0], sti.shape[1], 40, sti.shape[3]),
    #                          dtype=np.float16)
    #    L = sti[..., 0, :].copy()
    #    for j in range(L.shape[2]):
    #        x = L[..., j] / 100 * 255
    #        gabor_feat[..., j] = get_gabor_features(x)
    #    ofile = os.path.join(feat_dir, data_type+'_gabor.npy')
    #    np.save(ofile, gabor_feat)

    # features to expected BOLD signal
    #feat2bold(feat_dir, dataset='val', ftype='hue')

    # gaussian kernel based receptive field model
    #get_candidate_model(feat_dir)

    # pRF model fitting
    subj_id = 1
    roi = 'v1lh'
    # directory config
    subj_dir = os.path.join(db_dir, 'vS%s'%(subj_id))
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_fmri_ts(subj_dir, roi=roi)
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    train_models = np.load(os.path.join(feat_dir, 'train_candidate_model.npy'),
                           mmap_mode='r')
    val_models = np.load(os.path.join(feat_dir, 'val_candidate_model.npy'),
                         mmap_mode='r')
    prf_dir = os.path.join(subj_dir, 'prf')
    check_path(prf_dir)
    
    # lasso regression
    #paras_file = os.path.join(prf_dir, 'lassoreg_paras.npy')
    #paras = np.memmap(paras_file, dtype='float16', mode='w+',
    #                  shape=(len(vxl_idx), 15360, 46))
    #val_corr_file = os.path.join(prf_dir, 'lassoreg_pred_corr.npy')
    #val_corr = np.memmap(val_corr_file, dtype='float16', mode='w+',
    #                     shape=(len(vxl_idx), 15360))
    #alphas_file = os.path.join(prf_dir, 'lassoreg_alphas.npy')
    #alphas = np.memmap(alphas_file, dtype='float16', mode='w+',
    #                   shape=(len(vxl_idx), 15360))
    ## for code test
    #vxl_idx = vxl_idx[:5]
    #for i in range(len(vxl_idx)):
    #    print 'Voxel %s'%(i)
    #    train_y = train_fmri_ts[i]
    #    val_y = val_fmri_ts[i]
    #    for j in range(15360):
    #        print 'Model %s'%(j)
    #        train_x = np.array(train_models[j, ...]).astype(np.float64)
    #        val_x = np.array(val_models[j, ...]).astype(np.float64)
    #        train_x = zscore(train_x).T
    #        val_x = zscore(val_x).T
    #        lasso_cv = LassoCV(cv=10, n_jobs=6)
    #        lasso_cv.fit(train_x, train_y)
    #        alphas[i, j] = lasso_cv.alpha_
    #        paras[i, j :] = lasso_cv.coef_
    #        pred_y = lasso_cv.predict(val_x)
    #        val_corr[i, j] = np.corrcoef(val_y, pred_y)[0][1]
    #        print 'Alpha %s, prediction score %s'%(alphas[i, j], val_corr[i, j])
    #paras = np.array(paras)
    #np.save(paras_file, paras)
    #val_corr = np.array(val_corr)
    #np.save(val_corr_file, val_corr)
    #alphas = np.array(alphas)
    #np.save(alphas_file, alphas)
    
    # ridge regression
    ALPHA_NUM = 20
    BOOTS_NUM = 15
    paras_file = os.path.join(prf_dir, 'ridgereg_paras.npy')
    paras = np.memmap(paras_file, dtype='float64', mode='w+',
                      shape=(15360, len(vxl_idx), 46))
    val_corr_file= os.path.join(prf_dir, 'ridgereg_pred_corr.npy')
    val_corr = np.memmap(val_corr_file, dtype='float64', mode='w+',
                         shape=(15360, len(vxl_idx)))
    alphas_file = os.path.join(prf_dir, 'ridgereg_alphas.npy')
    alphas = np.memmap(alphas_file, dtype='float64', mode='w+',
                       shape=(15360, len(vxl_idx)))
    # fMRI data z-score
    print 'fmri data temporal z-score'
    m = np.mean(train_fmri_ts, axis=1, keepdims=True)
    s = np.std(train_fmri_ts, axis=1, keepdims=True)
    train_fmri_ts = (train_fmri_ts - m) / (1e-10 + s)
    m = np.mean(val_fmri_ts, axis=1, keepdims=True)
    s = np.std(val_fmri_ts, axis=1, keepdims=True)
    val_fmri_ts = (val_fmri_ts - m) / (1e-10 + s)

    # split training dataset into model tunning set and model selection set
    tune_fmri_ts = [:, :(7200*0.9)]
    sel_fmri_ts = [:, (7200*0.9):]

    for i in range(15360):
        print 'Model %s'%(i)
        train_x = np.array(train_models[i, ...]).astype(np.float64)
        train_x = zscore(train_x).T
        #val_x = np.array(val_models[i, ...]).astype(np.float64)
        #val_x = zscore(val_x).T
        # split training dataset into model tunning and selection sets
        tune_x = train_x[:7200*0.9, :]
        sel_x = train_x[7200*0.9:, :]
        wt, vcorr, alpha, bscores, valinds = ridge.bootstrap_ridge(
                tune_x, tune_fmri_ts.T, sel_x, sel_fmri_ts.T,
                alphas=np.logspace(-2, 3, ALPHA_NUM),
                nboots=BOOTS_NUM, chunklen=720, nchunks=1,
                single_alpha=False, use_corr=False)
        paras[i, ...] = wt.T
        val_corr[i] = vcorr
        alphas[i] = alpha
    # save output
    paras = np.array(paras)
    np.save(paras_file, paras)
    val_corr = np.array(val_corr)
    np.save(val_corr_file, val_corr)
    alphas = np.array(alphas)
    np.save(alphas_file, alphas)


