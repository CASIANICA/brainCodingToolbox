# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy.misc import imsave
import scipy.optimize as opt
from joblib import Parallel, delayed
import bob.ip.gabor
import bob.sp

from braincode.util import configParser
from braincode.math import ipl, make_2d_gaussian, ridge, make_cycle
from braincode.pipeline import retinotopy
from braincode.timeseries import hrf
from braincode.math.norm import zscore
from braincode.prf import dataio
from braincode.prf import util as vutil


def check_path(dir_path):
    """Check whether the directory does exist, if not, create it."""            
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, 0755)

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

def get_candidate_model(feat_dir, kernel='gaussian'):
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
    # candidate pooling fields included 15 evenly-spaces radii between 0.16
    # degrees (1 pixel) and 7.8 degrees (50 pixels)
    if kernel == 'round':
        feat_dir = os.path.join(feat_dir, 'round')
        check_path(feat_dir)
    out_train = os.path.join(feat_dir, 'train_candidate_model.npy')
    out_val = os.path.join(feat_dir, 'val_candidate_model.npy')
    train_model = np.memmap(out_train, dtype='float16', mode='w+',
                            shape=(32*32*15, 46, 7200))
    val_model = np.memmap(out_val, dtype='float16', mode='w+',
                          shape=(32*32*15, 46, 540))
    Parallel(n_jobs=4)(delayed(model_pro)(train_feat, val_feat, train_model,
                                          val_model, kernel, xi, yi, si)
                    for si in range(15) for xi in range(32) for yi in range(32))
    
    # save memmap object as a numpy.array
    train_array = np.array(train_model)
    np.save(out_train, train_array)
    val_array = np.array(val_model)
    np.save(out_val, val_array)

def model_pro(train_in, val_in, train_out, val_out, kernel, xi, yi, si):
    """Sugar function for generating candidate model"""
    mi = si*32*32 + xi*32 + yi
    center_x = np.arange(0, 128, 4)
    center_y = np.arange(0, 128, 4)
    sigma = np.linspace(1, 50, 15)
    x0 = center_x[xi]
    y0 = center_y[yi]
    s = sigma[si]
    print 'Model %s : center - (%s, %s), sigma %s'%(mi, x0, y0, s)
    if kernel == 'gaussian':
        kernel = make_2d_gaussian(128, s, center=(x0, y0))
    else:
        kernel = 0.01 * make_cycle(128, s, center=(x0, y0))
    kernel = kernel.flatten()
    tmp = np.zeros((331200, ), dtype=np.float16)
    for i in range(23):
        m = i*14400
        n = m + 14400
        tmp[m:n] = kernel.dot(train_in[:, m:n]).astype(np.float16)
    train_out[mi] = tmp.reshape(46, 7200)
    val_out[mi] = kernel.dot(val_in).reshape(46, 540).astype(np.float16)

def para2hue(paras):
    """Convert hue parameters to hue curve."""
    hues = np.zeros(201)
    for i in range(6):
        x = np.linspace(0, 2*np.pi, 201)
        tmp = np.sin(x - i*np.pi/3)
        tmp[tmp<0] = 0
        tmp = np.square(tmp)
        hues += paras[i]*tmp
    return hues

def ridge_fitting(feat_dir, prf_dir, db_dir, subj_id, roi):
    """pRF model fitting using ridge regression.
    90% trainning data used for model tuning, and another 10% data used for
    model seletion.
    """
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)
    del val_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    train_models = np.load(os.path.join(feat_dir, 'train_candidate_model.npy'),
                           mmap_mode='r')
    # output directory config
    roi_dir = os.path.join(prf_dir,  roi)
    check_path(roi_dir)
    # model seletion and tuning
    ALPHA_NUM = 20
    BOOTS_NUM = 15
    paras_file = os.path.join(roi_dir, 'reg_paras.npy')
    paras = np.memmap(paras_file, dtype='float64', mode='w+',
                      shape=(15360, len(vxl_idx), 46))
    mcorr_file= os.path.join(roi_dir, 'reg_model_corr.npy')
    mcorr = np.memmap(mcorr_file, dtype='float64', mode='w+',
                      shape=(15360, len(vxl_idx)))
    alphas_file = os.path.join(roi_dir, 'reg_alphas.npy')
    alphas = np.memmap(alphas_file, dtype='float64', mode='w+',
                       shape=(15360, len(vxl_idx)))
    # fMRI data z-score
    print 'fmri data temporal z-score'
    m = np.mean(train_fmri_ts, axis=1, keepdims=True)
    s = np.std(train_fmri_ts, axis=1, keepdims=True)
    train_fmri_ts = (train_fmri_ts - m) / (1e-10 + s)
    # split training dataset into model tunning set and model selection set
    tune_fmri_ts = train_fmri_ts[:, :int(7200*0.9)]
    sel_fmri_ts = train_fmri_ts[:, int(7200*0.9):]
    # model testing
    for i in range(15360):
        print 'Model %s'%(i)
        train_x = np.array(train_models[i, ...]).astype(np.float64)
        train_x = zscore(train_x).T
        # split training dataset into model tunning and selection sets
        tune_x = train_x[:int(7200*0.9), :]
        sel_x = train_x[int(7200*0.9):, :]
        wt, r, alpha, bscores, valinds = ridge.bootstrap_ridge(
                tune_x, tune_fmri_ts.T, sel_x, sel_fmri_ts.T,
                alphas=np.logspace(-2, 3, ALPHA_NUM),
                nboots=BOOTS_NUM, chunklen=720, nchunks=1,
                single_alpha=False, use_corr=False)
        paras[i, ...] = wt.T
        mcorr[i] = r
        alphas[i] = alpha
    # save output
    paras = np.array(paras)
    np.save(paras_file, paras)
    mcorr = np.array(mcorr)
    np.save(mcorr_file, mcorr)
    alphas = np.array(alphas)
    np.save(alphas_file, alphas)

def pls_ridge_fitting(feat_dir, prf_dir, db_dir, subj_id, roi):
    """pRF model fitting using ridge regression.
    90% trainning data used for model tuning, and another 10% data used for
    model seletion.
    """
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id)
    del val_fmri_ts
    roi_vxl_idx, roi_tfmri_ts, roi_vfmri_ts = dataio.load_vim2_fmri(db_dir,
                                                            subj_id, roi=roi)
    del roi_tfmri_ts
    del roi_vfmri_ts
    sel_idx = [i for i in range(vxl_idx.shape[0]) if vxl_idx[i] in roi_vxl_idx]
    pls_pred_fmri = np.load(os.path.join(prf_dir, 'pls_pred_tfmri_c20.npy'))
    d = train_fmri_ts - pls_pred_fmri.T
    train_fmri_ts = d[sel_idx]
    print train_fmri_ts.shape

    print 'Voxel number: %s'%(len(roi_vxl_idx))
    # load candidate models
    train_models = np.load(os.path.join(feat_dir, 'train_candidate_model.npy'),
                           mmap_mode='r')
    # output directory config
    roi_dir = os.path.join(prf_dir,  roi)
    check_path(roi_dir)
    # model seletion and tuning
    ALPHA_NUM = 20
    BOOTS_NUM = 15
    paras_file = os.path.join(roi_dir, 'reg_paras.npy')
    paras = np.memmap(paras_file, dtype='float64', mode='w+',
                      shape=(15360, len(sel_idx), 46))
    mcorr_file= os.path.join(roi_dir, 'reg_model_corr.npy')
    mcorr = np.memmap(mcorr_file, dtype='float64', mode='w+',
                      shape=(15360, len(sel_idx)))
    alphas_file = os.path.join(roi_dir, 'reg_alphas.npy')
    alphas = np.memmap(alphas_file, dtype='float64', mode='w+',
                       shape=(15360, len(sel_idx)))
    # fMRI data z-score
    print 'fmri data temporal z-score'
    m = np.mean(train_fmri_ts, axis=1, keepdims=True)
    s = np.std(train_fmri_ts, axis=1, keepdims=True)
    train_fmri_ts = (train_fmri_ts - m) / (1e-10 + s)
    # split training dataset into model tunning set and model selection set
    tune_fmri_ts = train_fmri_ts[:, :int(7200*0.9)]
    sel_fmri_ts = train_fmri_ts[:, int(7200*0.9):]
    # model testing
    for i in range(15360):
        print 'Model %s'%(i)
        train_x = np.array(train_models[i, ...]).astype(np.float64)
        train_x = zscore(train_x).T
        # split training dataset into model tunning and selection sets
        tune_x = train_x[:int(7200*0.9), :]
        sel_x = train_x[int(7200*0.9):, :]
        wt, r, alpha, bscores, valinds = ridge.bootstrap_ridge(
                tune_x, tune_fmri_ts.T, sel_x, sel_fmri_ts.T,
                alphas=np.logspace(-2, 3, ALPHA_NUM),
                nboots=BOOTS_NUM, chunklen=720, nchunks=1,
                single_alpha=False, use_corr=False)
        paras[i, ...] = wt.T
        mcorr[i] = r
        alphas[i] = alpha
    # save output
    paras = np.array(paras)
    np.save(paras_file, paras)
    mcorr = np.array(mcorr)
    np.save(mcorr_file, mcorr)
    alphas = np.array(alphas)
    np.save(alphas_file, alphas)

def prf_selection(feat_dir, prf_dir, db_dir, subj_id, roi):
    """Select best model for each voxel and validating."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    val_models = np.load(os.path.join(feat_dir, 'val_candidate_model.npy'),
                         mmap_mode='r')
    # output directory config
    roi_dir = os.path.join(prf_dir, roi)
    # load candidate model parameters 
    paras = np.load(os.path.join(roi_dir, 'reg_paras.npy'))
    mcorr = np.load(os.path.join(roi_dir, 'reg_model_corr.npy'))
    alphas = np.load(os.path.join(roi_dir, 'reg_alphas.npy'))
    sel_paras = np.zeros((mcorr.shape[1], 46))
    sel_model = np.zeros(mcorr.shape[1])
    sel_model_corr = np.zeros(mcorr.shape[1])
    for i in range(mcorr.shape[1]):
        maxi = np.argmax(mcorr[:, i])
        print 'Voxel %s - Max corr %s - Model %s'%(i, mcorr[maxi, i], maxi)
        print 'Alpha : %s'%(alphas[maxi, i])
        sel_paras[i] = paras[maxi, i]
        sel_model[i] = maxi
        feats = np.array(val_models[maxi, ...]).astype(np.float64)
        feats = zscore(feats).T
        pred = np.dot(feats, sel_paras[i])
        sel_model_corr[i] = np.corrcoef(pred, val_fmri_ts[i])[0, 1]
        print 'Val Corr : %s'%(sel_model_corr[i])
    np.save(os.path.join(roi_dir, 'reg_sel_paras.npy'), sel_paras)
    np.save(os.path.join(roi_dir, 'reg_sel_model.npy'), sel_model)
    np.save(os.path.join(roi_dir, 'reg_sel_model_corr.npy'), sel_model_corr)

def null_distribution_prf_tunning(feat_dir, prf_dir, db_dir, subj_id, roi):
    """Generate Null distribution of pRF model tunning using validation data."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    val_models = np.load(os.path.join(feat_dir, 'val_candidate_model.npy'),
                         mmap_mode='r')
    # output directory config
    roi_dir = os.path.join(prf_dir, roi)
    # load selected model parameters
    paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    null_corr = np.zeros((paras.shape[0], 1000))
    for i in range(paras.shape[0]):
        print 'Voxel %s'%(i)
        # load features
        feats = np.array(val_models[int(sel_model[i]), ...]).astype(np.float64)
        feats = zscore(feats).T
        pred = np.dot(feats, paras[i])
        for j in range(1000):
            shuffled_val_ts = np.random.permutation(val_fmri_ts[i])
            null_corr[i, j] = np.corrcoef(pred, shuffled_val_ts)[0, 1]
    np.save(os.path.join(roi_dir, 'random_corr.npy'), null_corr)

def gabor_contribution2prf(feat_dir, prf_dir, db_dir, subj_id, roi):
    """Calculate tunning contribution of each gabor sub-banks."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    val_models = np.load(os.path.join(feat_dir, 'val_candidate_model.npy'),
                         mmap_mode='r')
    # load selected model parameters
    roi_dir = os.path.join(prf_dir, roi)
    paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    gabor_corr = np.zeros((paras.shape[0], 5))
    for i in range(paras.shape[0]):
        print 'Voxel %s'%(i)
        # load features
        feats = np.array(val_models[int(sel_model[i]), ...]).astype(np.float64)
        feats = zscore(feats).T
        for j in range(5):
            pred = np.dot(feats[:, (j*8):(j*8+8)], paras[i, (j*8):(j*8+8)])
            gabor_corr[i, j] = np.corrcoef(pred, val_fmri_ts[i])[0, 1]
    np.save(os.path.join(roi_dir, 'gabor_contributes.npy'), gabor_corr)

def prf_recon(prf_dir, db_dir, subj_id, roi):
    """Reconstruct pRF based on selected model."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    del val_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # output directory config
    roi_dir = os.path.join(prf_dir, roi)
    # pRF estimate
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    prfs = np.zeros((sel_models.shape[0], 128, 128))
    fig_dir = os.path.join(roi_dir, 'figs')
    check_path(fig_dir)
    for i in range(sel_models.shape[0]):
        # get pRF
        print 'Voxel %s, Val Corr %s'%(i, sel_model_corr[i])
        model_idx = int(sel_models[i])
        # get gaussian pooling field parameters
        si = model_idx / 1024
        xi = (model_idx % 1024) / 32
        yi = (model_idx % 1024) % 32
        x0 = np.arange(0, 128, 4)[xi]
        y0 = np.arange(0, 128, 4)[yi]
        s = np.linspace(1, 50, 15)[si]
        #kernel = make_cycle(128, s, center=(x0, y0))
        kernel = make_2d_gaussian(128, s, center=(x0, y0))
        kpos = np.nonzero(kernel)
        paras = sel_paras[i]
        for f in range(5):
            fwt = np.sum(paras[(f*8):(f*8+8)])
            fs = np.sqrt(2)**f*4
            for p in range(kpos[0].shape[0]):
                tmp = make_2d_gaussian(128, fs, center=(kpos[1][p],
                                                        kpos[0][p]))
                prfs[i] += fwt * kernel[kpos[0][p], kpos[1][p]] * tmp
        if sel_model_corr[i]>=0.25:
            prf_file = os.path.join(fig_dir,'Voxel_%s_%s.png'%(i+1, vxl_idx[i]))
            vutil.save_imshow(prfs[i], prf_file)
    np.save(os.path.join(roi_dir, 'prfs.npy'), prfs)

def filter_recon(prf_dir, db_dir, subj_id, roi):
    """Reconstruct filter map of each voxel based on selected model."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    del val_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # output config
    roi_dir = os.path.join(prf_dir, roi)
    # pRF estimate
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    filters = np.zeros((sel_models.shape[0], 128, 128))
    fig_dir = os.path.join(roi_dir, 'filters')
    check_path(fig_dir)
    thr = 0.17
    # gabor bank generation
    gwt = bob.ip.gabor.Transform()
    gwt.generate_wavelets(128, 128)
    spatial_gabors = np.zeros((40, 128, 128))
    for i in range(40):
        w = bob.ip.gabor.Wavelet(resolution=(128, 128),
                        frequency=gwt.wavelet_frequencies[i])
        sw = bob.sp.ifft(w.wavelet.astype(np.complex128)) 
        spatial_gabors[i, ...] = np.roll(np.roll(np.real(sw), 64, 0), 64, 1)
    for i in range(sel_models.shape[0]):
        if sel_model_corr[i]<thr:
            continue
        print 'Voxel %s, Val Corr %s'%(i, sel_model_corr[i])
        model_idx = int(sel_models[i])
        # get gaussian pooling field parameters
        si = model_idx / 1024
        xi = (model_idx % 1024) / 32
        yi = (model_idx % 1024) % 32
        x0 = np.arange(0, 128, 4)[xi]
        y0 = np.arange(0, 128, 4)[yi]
        s = np.linspace(1, 50, 15)[si]
        kernel = make_2d_gaussian(128, s, center=(x0, y0))
        kpos = np.nonzero(kernel)
        paras = sel_paras[i]
        for gwt_idx in range(40):
            wt = paras[gwt_idx]
            arsw = spatial_gabors[gwt_idx]
            for p in range(kpos[0].shape[0]):
                tmp = img_offset(arsw, (kpos[0][p], kpos[1][p]))
                filters[i] += wt * kernel[kpos[0][p], kpos[1][p]] * tmp
        if sel_model_corr[i]>=thr:
            im_file = os.path.join(fig_dir, 'Voxel_%s_%s.png'%(i+1, vxl_idx[i]))
            vutil.save_imshow(filters[i], im_file)
    np.save(os.path.join(roi_dir, 'filters.npy'), filters)

def stimuli_recon(prf_dir, db_dir, subj_id, roi):
    """Reconstruct stimulus based on pRF model."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load model parameters
    roi_dir = os.path.join(prf_dir, roi)
    val_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    filters = np.load(os.path.join(roi_dir, 'filters.npy'))
    recon_imgs = np.zeros((val_fmri_ts.shape[1], 128, 128))
    # fMRI data z-score
    print 'fmri data temporal z-score'
    m = np.mean(val_fmri_ts, axis=1, keepdims=True)
    s = np.std(val_fmri_ts, axis=1, keepdims=True)
    val_fmri_ts = (val_fmri_ts - m) / (1e-10 + s)
    # select significant predicted voxels
    sel_vxls = np.nonzero(val_corr>=0.17)[0]
    for i in range(val_fmri_ts.shape[1]):
        print 'Reconstruct stimilus %s'%(i+1)
        tmp = np.zeros((128, 128))
        for j in sel_vxls:
            tmp += val_fmri_ts[int(j), int(i)] * filters[j]
        recon_imgs[i] = tmp
    np.save(os.path.join(roi_dir, 'recon_img.npy'), recon_imgs)

def get_predicted_fmri(feat_dir, prf_dir, roi, data_type):
    """Get estimated fmri responses based on encoding model."""
    roi_dir = os.path.join(prf_dir, roi)
    # load model parameters
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    # load candidate model
    model_file = os.path.join(feat_dir, '%s_candidate_model.npy'%(data_type))
    models = np.load(model_file, mmap_mode='r')
    # output
    pred_fmri = np.zeros((sel_paras.shape[0], models.shape[2]))
    for i in range(sel_paras.shape[0]):
        print 'Voxel %s'%(i)
        model_idx = int(sel_models[i])
        x = np.array(models[model_idx, ...]).astype(np.float64)
        m = np.mean(x, axis=0, keepdims=True)
        s = np.std(x, axis=0, keepdims=True)
        x = (x - m) / (s + 1e-5)
        wts = sel_paras[i].reshape(1, -1)
        pred_fmri[i] = np.dot(wts, x)
    outfile = os.path.join(roi_dir, '%s_pred_norm_fmri.npy'%(data_type))
    np.save(outfile, pred_fmri)

def get_prediction_residual(prf_dir, db_dir, subj_id):
    roi_list = ['v1rh', 'v1lh', 'v2rh', 'v2lh', 'v3rh', 'v3lh', 'v4rh', 'v4lh']
    orig_fmri = None
    pred_fmri = None
    res_fmri = None
    vxl_idx = None
    for roi in roi_list:
        idx, tx, vx = dataio.load_vim2_fmri(db_dir, subj_id, roi)
        m = tx.mean(axis=1, keepdims=True)
        s = tx.mean(axis=1, keepdims=True)
        roi_pred_file = os.path.join(prf_dir, roi, 'train_pred_norm_fmri.npy')
        roi_pred_fmri = np.load(roi_pred_file)
        roi_pred_fmri = roi_pred_fmri * s + m
        res = tx - roi_pred_fmri
        if not isinstance(res_fmri, np.ndarray):
            orig_fmri = tx
            pred_fmri = roi_pred_fmri
            res_fmri = res
            vxl_idx = idx
        else:
            orig_fmri = np.vstack((orig_fmri, tx))
            pred_fmri = np.vstack((pred_fmri, roi_pred_fmri))
            res_fmri = np.vstack((res_fmri, res))
            vxl_idx = np.concatenate((vxl_idx, idx))
    outfile = os.path.join(prf_dir, 'roi_coding_fmri')
    np.savez(outfile,orig_fmri=orig_fmri, pred_fmri=pred_fmri,
             res_fmri=res_fmri, vxl_idx=vxl_idx)

def get_hue_selectivity(prf_dir, db_dir, subj_id, roi):
    """Get hue tunning curve for each voxel and calculate hue selectivity."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    del val_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # pRF estimate
    roi_dir = os.path.join(prf_dir, roi)
    sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    hue_tunes = np.zeros((len(vxl_idx), 201))
    hue_sel = np.zeros(len(vxl_idx))
    for i in range(len(vxl_idx)):
        print 'Voxel %s, Val Corr %s'%(i, sel_model_corr[i])
        paras = sel_paras[i]
        # get hue selection
        hue_tunes[i] = para2hue(paras[40:])
        hue_sel[i] = abs(hue_tunes[i].max()-hue_tunes[i].min())
        if sel_model_corr[i]>=0.25:
            hue_file = os.path.join(roi_dir,
                                    'Voxel_%s_%s_hue.png'%(i+1, vxl_idx[i]))
            vutil.save_hue(hue_tunes[i], hue_file)
    np.save(os.path.join(roi_dir, 'hue_tunes.npy'), hue_tunes)
    np.save(os.path.join(roi_dir, 'hue_selectivity.npy'), hue_sel)

def retinotopic_mapping(prf_dir, roi):
    """Get eccentricity and angle based on pRF center for each voxel."""
    roi_dir = os.path.join(prf_dir, roi)
    # load selected model index
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    # output variables
    ecc = np.zeros_like(sel_models)
    angle = np.zeros_like(sel_models)
    coords = np.zeros((sel_models.shape[0], 2))
    for i in range(sel_models.shape[0]):
        print 'Voxel %s'%(i+1)
        model_idx = int(sel_models[i])
        xi = (model_idx % 1024) / 32
        yi = (model_idx % 1024) % 32
        # col
        x0 = np.arange(0, 128, 4)[xi]
        # row
        y0 = np.arange(0, 128, 4)[yi]
        coords[i] = [y0, x0]
    # get ecc and angle
    ecc = retinotopy.coord2ecc(coords, 128, 20)
    angle = retinotopy.coord2angle(coords, 128)
    np.save(os.path.join(roi_dir, 'ecc.npy'), ecc)
    np.save(os.path.join(roi_dir, 'angle.npy'), angle)

def curve_fit(prf_dir, roi):
    """Get pRF parameters using model fitting based on various kernels."""
    roi_dir = os.path.join(prf_dir, roi)
    # load pRF maps and the selected model index
    prfs = np.load(os.path.join(roi_dir, 'reg_prfs.npy'))
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    # last column is curve fitting error based on squared-differnece
    paras = np.zeros((sel_models.shape[0], 6))
    for i in range(sel_models.shape[0]):
        if np.sum(prfs[i])==0:
            paras[i, :] = np.nan
            continue
        # get prf parameters
        print 'Voxel %s'%(i+1)
        model_idx = int(sel_models[i])
        xi = (model_idx % 1024) / 32
        yi = (model_idx % 1024) % 32
        x0 = np.arange(0, 128, 4)[xi]
        y0 = np.arange(0, 128, 4)[yi]
        initial_guess = (y0, x0, 1, 0, 1)
        try:
            y = prfs[i].flatten()
            popt, pcov = opt.curve_fit(vutil.sugar_gaussian_f, 128, y,
                                       p0=initial_guess)
            #print popt
            paras[i, :5] = popt
            pred_y = vutil.sugar_gaussian_f(128, *popt)
            paras[i, 5] = np.square(y-pred_y).sum()
        except RuntimeError:
            print 'Error - curve_fit failed'
            paras[i, :] = np.nan
    np.save(os.path.join(roi_dir, 'prf_curve_fit_paras.npy'), paras)

def show_retinotopy(prf_dir, data_type):
    """Show estimate parameters on brain."""
    #data_type = 'ecc'
    vol = np.empty((73728,))
    vol[:] = np.NaN
    subj_dir = '/'.join(prf_dir.split('/')[:-2])
    # get roi list
    roi_list = ['v1lh', 'v2lh', 'v3lh', 'v4lh',
                'v1rh', 'v2rh', 'v3rh', 'v4rh']
    rois = os.listdir(prf_dir)
    rois = [roi for roi in rois if roi in roi_list]
    for roi in rois:
        vxl_idx, train_ts, val_ts = dataio.load_fmri(subj_dir, roi=roi)
        del train_ts, val_ts
        data = np.load(os.path.join(prf_dir, roi, data_type+'.npy'))
        vol[vxl_idx] = data
    vol = vol.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(prf_dir, data_type+'.nii.gz'))

def img_offset(orig_img, new_center):
    """Move original image to new position based on new center coordinate.
    new_center is (x0, y0), x0 indicates row coordinate, y0 indicates col
    coordinate.
    """
    img_r, img_c = orig_img.shape
    new_img = np.zeros_like(orig_img)
    # move image position based on new center coordinates
    old_x0 = img_r // 2
    old_y0 = img_c // 2
    offset0 = int(np.rint(new_center[0] - old_x0))
    offset1 = int(np.rint(new_center[1] - old_y0))
    pixs = np.mgrid[0:img_r, 0:img_c].reshape(2, img_r*img_c)
    new_x = pixs[0] + offset0
    new_y = pixs[1] + offset1
    pix_idx = (new_x>=0) * (new_x<img_r) * (new_y>=0) * (new_y<img_c)
    new_img[new_x[pix_idx], new_y[pix_idx]] = orig_img[pixs[0, pix_idx],
                                                       pixs[1, pix_idx]]
    return new_img


if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    # directory config for database
    db_dir = cf.get('database', 'path')
    db_dir = os.path.join(db_dir, 'vim2')
    # directory config for analysis
    root_dir = cf.get('base', 'path')
    stim_dir = os.path.join(root_dir, 'stimulus', 'vim2')
    feat_dir = os.path.join(root_dir, 'sfeatures', 'vim2')
    res_dir = os.path.join(root_dir, 'subjects')

    #-- load original stimulus data
    #tf = tables.open_file(os.path.join(db_dir, 'Stimuli.mat'))
    #data_type = 'train'
    #data_dic = {'train': '/st', 'val': '/sv'}
    #stimulus = tf.get_node(data_dic[data_type])[:]
    #tf.close()
    
    #-- convert mat to png
    #mat2png(stimulus, data_type)
    
    #-- convert images from RGB to CIRLCh space
    #stimulus = np.transpose(stimulus, (3, 2, 1, 0))
    #ciesti = np.zeros_like(stimulus, dtype=np.float16)
    #for i in range(stimulus.shape[3]):
    #    tmp = ipl.rgb2cielab(stimulus[..., i])
    #    ciesti[..., i] = ipl.cielab2cielch(tmp)
    #np.save(os.path.join(stim_dir, data_type+'_stimulus_LCh.npy'), ciesti)
    
    #-- extract hue features
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

    #-- extract gabor features
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

    #-- features to expected BOLD signal
    #feat2bold(feat_dir, dataset='val', ftype='hue')

    #-- gaussian kernel based receptive field model
    #get_candidate_model(feat_dir, kernel='gaussian')
    #get_candidate_model(feat_dir, kernel='round')

    #-- general config
    subj_id = 1
    kernel = 'gaussian'
    roi = 'v1lh'
    # directory config
    if kernel=='round':
        feat_dir = os.path.join(feat_dir, 'round')
    subj_dir = os.path.join(res_dir, 'vim2_S%s'%(subj_id))
    prf_dir = os.path.join(subj_dir, 'prf', kernel+'_kernel')

    #-- pRF model fitting
    # pRF model tuning
    #ridge_fitting(feat_dir, prf_dir, db_dir, subj_id, roi)
    # pRF model selection and validation
    #prf_selection(feat_dir, prf_dir, db_dir, subj_id, roi)
    # get null distribution of pRF tunning
    #null_distribution_prf_tunning(feat_dir, prf_dir, db_dir, subj_id, roi)
    # pRF reconstruction
    #prf_recon(prf_dir, db_dir, subj_id, roi)
    # filter reconstruction
    #filter_recon(prf_dir, db_dir, subj_id, roi)
    # validation stimuli reconstruction
    #stimuli_recon(prf_dir, db_dir, subj_id, roi)
    # get hue selectivity for each voxel
    #get_hue_selectivity(prf_dir, db_dir, subj_id, roi)
    # get eccentricity and angle based on pRF center for each voxel
    #retinotopic_mapping(prf_dir, roi)
    # get pRF parameters using curve-fitting
    #curve_fit(roi_dir)
    # calculate tunning contribution of each gabor sub-banks
    #gabor_contribution2prf(feat_dir, prf_dir, db_dir, subj_id, roi)

    #-- show retinotopic mapping
    #show_retinotopy(prf_dir, 'ecc')

    #prf_dir = os.path.join(subj_dir, 'pls')
    #pls_ridge_fitting(feat_dir, prf_dir, db_dir, subj_id, roi)

    # Get estimated brain activity based on encoding model
    #get_predicted_fmri(feat_dir, prf_dir, roi, 'train')
    # Get predicted fmri residual
    get_prediction_residual(prf_dir, db_dir, subj_id)

