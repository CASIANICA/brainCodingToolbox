# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
import bob.ip.gabor
from joblib import Parallel, delayed

from braincode.util import configParser
from braincode.math import make_2d_gaussian


def get_gabor_features(img):
    """Get Gabor features from input image."""
    img = img.astype(np.float64)
    gwt = bob.ip.gabor.Transform(number_of_scales=9)
    trafo_img = gwt(img)
    out_feat = np.zeros((72, 500, 500))
    for i in range(trafo_img.shape[0]):
        real_p = np.real(trafo_img[i, ...])
        imag_p = np.imag(trafo_img[i, ...])
        out_feat[i, ...] = np.sqrt(np.square(real_p)+np.square(imag_p))
    return out_feat

def get_stim_features(db_dir, feat_dir, data_type):
    """Stimuli processing."""
    sti_dir = os.path.join(db_dir, 'stimuli')
    prefix = {'train': 'Stimuli_Trn_FullRes', 'val': 'Stimuli_Val_FullRes'}
    if data_type == 'train':
        for i in range(15):
            mat_file = os.path.join(sti_dir, prefix['train']+'_%02d.mat'%(i+1))
            print 'Load file %s ...'%(mat_file)
            tf = tables.open_file(mat_file)
            imgs = tf.get_node('/stimTrn')[:]
            tf.close()
            # output matrix: image-number x channel x row x col
            print 'image size %s'%(imgs.shape[2])
            out_features = np.zeros((imgs.shape[2], 72, 500, 500))
            for j in range(imgs.shape[2]):
                x = imgs[..., j].T
                out_features[j, ...] = get_gabor_features(x)
            out_file = prefix['train']+'_%02d_gabor_features.npy'%(i+1)
            out_file = os.path.join(feat_dir, out_file)
            np.save(out_file, out_features)
    else:
        mat_file = os.path.join(sti_dir, prefix['val']+'.mat')
        print 'Load file %s ...'%(mat_file)
        tf = tables.open_file(mat_file)
        imgs = tf.get_node('/stimVal')[:]
        # output matrix: image-number x channel x row x col
        out_features = np.zeros((imgs.shape[2], 72, 500, 500))
        for j in range(imgs.shape[2]):
            x = imgs[..., j].T
            out_features[j, ...] = get_gabor_features(x)
        out_file = prefix['val']+'_gabor_features.npy'
        out_file = os.path.join(feat_dir, out_file)
        np.save(out_file, out_features)

def get_candidate_model(feat_dir, data_type):
    """Get gaussian kernel based on receptivefield features."""
    prefix = {'train': 'Stimuli_Trn_FullRes', 'val': 'Stimuli_Val_FullRes'}
    feat_ptr = []
    if data_type == 'train':
        time_count = 0
        for i in range(15):
            tmp_file = os.path.join(feat_dir,
                    prefix['train']+'_%02d_gabor_features.npy'%(i+1))
            tmp = np.load(tmp_file, mmap_mode='r')
            time_count += tmp.shape[0]
            feat_ptr.append(tmp)
    else:
        tmp_file = os.path.join(feat_dir, prefix['val']+'_gabor_features.npy')
        tmp = np.load(tmp_file, mmap_mode='r')
        time_count = tmp.shape[0]
        feat_ptr.append(tmp)
    print 'Time series length: %s'%(time_count)

    # derived gauusian-kernel based features
    # candidate pooling centers are spaces 0.4 degrees apart (10 pixels)
    # candidate pooling fields included 17 radii (1, 5, 10, 15, 20, ..., 
    # 55, 60, 70, 80, 90, 100 pixels) between 0.04 degree (1 pixel) and 4
    # degree (100 pixels)
    out_file = os.path.join(feat_dir, '%s_candidate_model.npy'%(data_type))
    cand_model = np.memmap(out_file, dtype='float16', model='w+',
                           shape=(50*50*17, time_count, 72))
    Parallel(n_jobs=4)(delayed(model_pro)(feat_ptr, cand_model, xi, yi, si)
                    for si in range(17) for xi in range(50) for yi in range(50))
    # save memmap object as a numpy.array
    model_array = np.array(cand_model)
    np.save(out_file, model_array)

def model_pro(feat_ptr, cand_model, xi, yi, si):
    """Sugar function for generating candidate model."""
    mi = si*50*50 + xi*50 + yi
    center_x = np.arange(5, 500, 10)
    center_y = np.arange(5, 500, 10)
    sigma = [1] + [n*5 for n in range(1, 13)] + [70, 80, 90, 100]
    x0 = center_x[xi]
    y0 = center_y[yi]
    s = sigma[si]
    print 'Model %s : center - (%s, %s), sigma %s'%(mi, y0, x0, s)
    kernel = make_2d_gaussian(500, s, center=(x0, y0))
    kernel = kernel.flatten()
    idx_head = 0
    for feat in feat_ptr:
        parts = feat.shape[0] / 10
        for i in range(parts):
            tmp = feat[idx_head:(idx_head+10), ...]
            tmp = tmp.reshape(720, 250000)
            res = tmp.dot(kernel).astype(np.float16)
            print res.max(), res.min()
            cand_model[mi, idx_head:(idx_head+10), ...] = res.reshape(10, 72)
            idx_head += 10

def stim_downsample():
    """Stimuli processing."""
    #db_dir = r'/home/huanglijie/workingdir/brainDecoding/vim1/gabor_features'
    #prefix = {'train': 'Stimuli_Trn_FullRes', 'val': 'Stimuli_Val_FullRes'}

    #data_type = 'val'

    #if data_type == 'train':
    #    for i in range(15):
    #        src_file = os.path.join(db_dir,
    #                        prefix['train']+'_%02d_gabor_features.npy'%(i+1))
    #        print 'Load file %s ...'%(src_file)
    #        imgs = np.load(src_file)
    #        # output matrix: row x col x image-number
    #        print 'image size %s'%(imgs.shape[3])
    #        out_imgs = imgs[..., 0:8, :].sum(axis=2)
    #        out_file = prefix['train']+'_%02d_smallest_gabor_features.npy'%(i+1)
    #        np.save(out_file, out_imgs)
    #else:
    #    src_file = os.path.join(db_dir, prefix['val']+'_gabor_features.npy')
    #    print 'Load file %s ...'%(src_file)
    #    imgs = np.load(src_file)
    #    # output matrix: row x col x image-number
    #    print 'image size %s'%(imgs.shape[3])
    #    out_imgs = imgs[..., 0:8, :].sum(axis=2)
    #    out_file = prefix['val']+'smallest_gabor_features.npy'
    #    np.save(out_file, out_imgs)
    pass

if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    # database directory config
    db_dir = os.path.join(cf.get('database', 'path'), 'vim1')
    # directory config for analysis
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures', 'vim1')
    res_dir = os.path.join(root_dir, 'subjects')
 
    # get gabor features
    #get_stim_features(db_dir, feat_dir, 'train')
    # get candidate models
    get_candidate_model(feat_dir, 'val')

