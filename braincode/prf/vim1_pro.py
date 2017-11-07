# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
import bob.ip.gabor

from braincode.util import configParser


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
    get_stim_features(db_dir, feat_dir, 'train')

