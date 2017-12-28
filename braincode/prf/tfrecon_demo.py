# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import tables
import tensorflow as tf
import matplotlib.pyplot as plt

def reconstructor(gabor_bank, vxl_coding_paras, y):
    """Stimuli reconstructor based on Activation Maximization"""
    # var for input stimuli
    img = tf.Variable(tf.random_normal([1, 500, 500, 1], stddev=0.001),
                      name="image")
    # config for the gabor filters
    gabor_real = np.expand_dims(gabor_bank['gabor_real'], 2)
    gabor_imag = np.expand_dims(gabor_bank['gabor_imag'], 2)
    real_conv = tf.nn.conv2d(img, gabor_real, strides=[1, 1, 1, 1],
                             padding='SAME')
    imag_conv = tf.nn.conv2d(img, gabor_imag, strides=[1, 1, 1, 1],
                             padding='SAME')
    gabor_energy = tf.sqrt(tf.square(real_conv) + tf.square(imag_conv))
    # reshape gabor energy for pRF masking
    gabor_vtr = tf.reshape(gabor_energy, [250000, 72])
    # weighted by voxel encoding models
    vxl_masks = vxl_coding_paras['masks']
    vxl_wts = vxl_coding_paras['wts']
    vxl_bias = vxl_coding_paras['bias']
    # masked by pooling fields
    vxl_masks = vxl_masks.reshape(-1, 250000)
    vxl_feats = tf.matmul(vxl_masks, gabor_vtr)
    vxl_wt_feats = tf.multiply(vxl_feats, vxl_wts)
    vxl_rsp = tf.reduce_sum(vxl_wt_feats, axis=1)
    vxl_pred = vxl_rsp - vxl_bias
    # input config
    vxl_real = tf.placeholder(tf.float32,
                shape=(vxl_coding_paras['bias'].shape[0],))
    error = tf.reduce_mean(tf.square(vxl_pred - vxl_real))
    opt = tf.train.GradientDescentOptimizer(0.5)
    vars_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "image")
    solver =  opt.minimize(error, var_list = vars_x)
 
    # training
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())     
    print y[:,2].shape
    
    for step in range(500):  
        _, error_curr, reconstructed_img = sess.run([solver, error, img], feed_dict={vxl_real: y[:, 2]}) 

        if step % 100 == 0:
            print('Iter: {}; loss: {:.4}'.format(step, error_curr))    
            fig=plt.figure()
            plt.imshow(reconstructed_img.reshape(500, 500))
            plt.savefig('recons'+str(step)+'.png')
            plt.close(fig)             
    return reconstructed_img


if __name__ == '__main__':
    """Main function"""
    # database directory config
    # directory config for analysis
    root_dir = r'/nfs/home/cddu/ActMax'
    db_dir = os.path.join(root_dir, 'db')
    res_dir = os.path.join(root_dir, 'subjects')

    #-- general config
    subj_id = 1
    roi = 'v1'
    # directory config
    subj_dir = os.path.join(res_dir, 'vim1_S%s'%(subj_id))
    prf_dir = os.path.join(subj_dir, 'prf')

    # parameter preparation
    gabor_bank_file = os.path.join(root_dir, 'gabor_kernels.npz')
    gabor_bank = np.load(gabor_bank_file)
    vxl_coding_paras_file = os.path.join(prf_dir, roi, 'tfrecon',
                                         'center_vxl_coding_wts.npz')
    vxl_coding_paras = np.load(vxl_coding_paras_file)

    #-- stimuli reconstruction
    resp_file = os.path.join(db_dir, 'EstimatedResponses.mat')
    resp_mat = tables.open_file(resp_file)
    # create mask
    # train data shape: (1750, ~25000)
    train_ts = resp_mat.get_node('/dataTrnS%s'%(subj_id))[:]
    train_ts = train_ts.T
    # get non-NaN voxel index
    fmri_s = train_ts.sum(axis=1)
    non_nan_idx = np.nonzero(np.logical_not(np.isnan(fmri_s)))[0]
    rois = resp_mat.get_node('/roiS%s'%(subj_id))[:]
    rois = rois[0]
    roi_idx = {'v1': 1, 'v2': 2, 'v3': 3, 'v3a': 4,
               'v3b': 5, 'v4': 6, 'LO': 7}
    vxl_idx = np.nonzero(rois==roi_idx[roi])[0]
    vxl_idx = np.intersect1d(vxl_idx, non_nan_idx)
    # load fmri response: data shape (#voxel, 1750/120)
    train_ts = np.nan_to_num(train_ts[vxl_idx])
    m = np.mean(train_ts, axis=1, keepdims=True)
    s = np.std(train_ts, axis=1, keepdims=True)
    train_ts = (train_ts - m) / (s + 1e-5)
    #val_ts = tf.get_node('/dataValS%s'%(subj_id))[:]
    #val_ts = val_ts.T
    #val_ts = np.nan_to_num(val_ts[vxl_idx])
    resp_mat.close()
    y_ = train_ts[vxl_coding_paras['vxl_idx']]
    # shape: (#voxel, 1750)
    print y_.shape
    recon_img = reconstructor(gabor_bank, vxl_coding_paras, y_)
 
    # show image    
    fig=plt.figure()
    plt.imshow(recon_img.reshape(500, 500))
    plt.savefig('recons.png')
    recon_img = recon_img.reshape(500, 500)

