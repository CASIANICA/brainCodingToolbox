# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tensorflow as tf

from braincode.util import configParser


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

def model_test(input_imgs, gabor_bank, vxl_coding_paras):
    """Stimuli reconstructor based on Activation Maximization"""
    # var for input stimuli
    img = tf.placeholder("float", shape=[None, 500, 500, 1])
    # config for the gabor filters
    gabor_real = np.expand_dims(gabor_bank['gabor_real'], 2)
    gabor_imag = np.expand_dims(gabor_bank['gabor_imag'], 2)
    real_conv = tf.nn.conv2d(img, gabor_real, strides=[1, 1, 1, 1],
                             padding='SAME')
    imag_conv = tf.nn.conv2d(img, gabor_imag, strides=[1, 1, 1, 1],
                             padding='SAME')
    gabor_energy = tf.sqrt(tf.square(real_conv) + tf.square(imag_conv))
    #gabor_pool = tf.nn.avg_pool(gabor_energy, ksize=[1, 2, 2, 1],
    #                            strides=[1, 2, 2, 1], padding='SAME')
    gabor_pool = tf.image.resize_images(gabor_energy, size=[250, 250])
    vxl_wts = vxl_coding_paras['wt']
    vxl_bias = vxl_coding_paras['bias']
    vxl_conv = tf.nn.conv2d(gabor_pool, vxl_wts, strides=[1, 1, 1, 1],
                            padding='VALID')
    vxl_conv = tf.reshape(vxl_conv, [-1])
    vxl_out = vxl_conv - vxl_bias
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(input_imgs.shape[2]):
            x = input_imgs[..., i].T
            x = np.expand_dims(x, 0)
            x = np.expand_dims(x, 3)
            resp = sess.run(vxl_out, feed_dict={img: x})
            print resp

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

    # test encoding model
    gabor_bank_file = os.path.join(feat_dir, 'gabor_kernels.npz')
    gabor_bank = np.load(gabor_bank_file)
    vxl_coding_paras_file = os.path.join(prf_dir, roi, 'tfrecon',
                                         'vxl_coding_wts.npz')
    vxl_coding_paras = np.load(vxl_coding_paras_file)
    img_file = os.path.join(root_dir, 'example_imgs.npy')
    imgs = np.load(img_file)
    model_test(imgs, gabor_bank, vxl_coding_paras)

