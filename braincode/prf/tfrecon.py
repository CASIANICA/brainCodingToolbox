# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tensorflow as tf

from braincode.util import configParser
from braincode.math import make_2d_gaussian


def reconstructor(gabor_bank, vxl_coding_paras):
    """Stimuli reconstructor based on Activation Maximization"""
    # var for input stimuli
    img = tf.Variable(tf.zeros([1, 500, 500, 1]))
    # config for the gabor filters
    gabor_real = np.expand_dims(gabor_bank['gabor_real'], 2)
    gabor_imag = np.expand_dims(gabor_bank['gabor_imag'], 2)
    real_conv = tf.nn.conv2d(img, gabor_real, strides=[1, 1, 1, 1],
                             padding='SAME')
    imag_conv = tf.nn.conv2d(img, gabor_imag, strides=[1, 1, 1, 1],
                             padding='SAME')
    gabor_energy = tf.sqrt(tf.square(real_conv) + tf.square(imag_conv))
    gabor_pool = tf.nn.max_pool(gabor_energy, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')
    vxl_wts = vxl_coding_paras['wt']
    vxl_bias = vxl_coding_paras['bias']
    vxl_conv = tf.nn.conv2d(gabor_pool, vxl_wts, strides=[1, 1, 1, 1],
                            padding='VALID')
    vxl_conv = tf.reshape(vxl_wts.shape[3])
    vxl_out = vxl_conv - bias



if __name__ == '__main__':
    """Main function"""
    # config parser
    cf = configParser.Config('config')
    # database directory config
    db_dir = os.path.join(cf.get('database', 'path'), 'vim1')
    # directory config for analysis
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures', 'vim1')
    res_dir = os.path.join(root_dir, 'subjects')

    #-- general config
    subj_id = 1
    roi = 'v1'
    # directory config
    subj_dir = os.path.join(res_dir, 'vim1_S%s'%(subj_id))
    prf_dir = os.path.join(subj_dir, 'prf')

    wt, bias = get_vxl_coding_wts(feat_dir, prf_dir, roi)
    np.savez('coding_paras', wt=wt, bias=bias)

