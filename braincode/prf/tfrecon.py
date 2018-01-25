# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os    
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np
import tables
import tensorflow as tf
import tensorflow.contrib.distributions as ds
import matplotlib.pyplot as plt

from braincode.util import configParser

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

def model_test(input_imgs, gabor_bank, vxl_coding_paras):
    """pRF encoding model tests."""
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
    vxl_out = vxl_rsp - vxl_bias
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(input_imgs.shape[2]):
            x = input_imgs[..., i].T
            x = np.expand_dims(x, 0)
            x = np.expand_dims(x, 3)
            resp = sess.run(vxl_out, feed_dict={img: x})
            print resp

def gabor_test(input_imgs, gabor_bank):
    """Test gabor wavelets for feature extraction."""
    # var for input stimuli
    img = tf.placeholder("float", shape=[None, 500, 500, 1])
    # config for the gabor filters
    f1_real = np.expand_dims(gabor_bank['f1_real'], 2)
    f1_imag = np.expand_dims(gabor_bank['f1_imag'], 2)
    f1_real_conv = tf.nn.conv2d(img, f1_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f1_imag_conv = tf.nn.conv2d(img, f1_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f1_energy = tf.sqrt(tf.square(f1_real_conv) + tf.square(f1_imag_conv))
    f2_real = np.expand_dims(gabor_bank['f2_real'], 2)
    f2_imag = np.expand_dims(gabor_bank['f2_imag'], 2)
    f2_real_conv = tf.nn.conv2d(img, f2_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f2_imag_conv = tf.nn.conv2d(img, f2_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f2_energy = tf.sqrt(tf.square(f2_real_conv) + tf.square(f2_imag_conv))
    f3_real = np.expand_dims(gabor_bank['f3_real'], 2)
    f3_imag = np.expand_dims(gabor_bank['f3_imag'], 2)
    f3_real_conv = tf.nn.conv2d(img, f3_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f3_imag_conv = tf.nn.conv2d(img, f3_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f3_energy = tf.sqrt(tf.square(f3_real_conv) + tf.square(f3_imag_conv))
    f4_real = np.expand_dims(gabor_bank['f4_real'], 2)
    f4_imag = np.expand_dims(gabor_bank['f4_imag'], 2)
    f4_real_conv = tf.nn.conv2d(img, f4_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f4_imag_conv = tf.nn.conv2d(img, f4_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f4_energy = tf.sqrt(tf.square(f4_real_conv) + tf.square(f4_imag_conv))
    f5_real = np.expand_dims(gabor_bank['f5_real'], 2)
    f5_imag = np.expand_dims(gabor_bank['f5_imag'], 2)
    f5_real_conv = tf.nn.conv2d(img, f5_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f5_imag_conv = tf.nn.conv2d(img, f5_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f5_energy = tf.sqrt(tf.square(f5_real_conv) + tf.square(f5_imag_conv))
    f6_real = np.expand_dims(gabor_bank['f6_real'], 2)
    f6_imag = np.expand_dims(gabor_bank['f6_imag'], 2)
    f6_real_conv = tf.nn.conv2d(img, f6_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f6_imag_conv = tf.nn.conv2d(img, f6_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f6_energy = tf.sqrt(tf.square(f6_real_conv) + tf.square(f6_imag_conv))
    f7_real = np.expand_dims(gabor_bank['f7_real'], 2)
    f7_imag = np.expand_dims(gabor_bank['f7_imag'], 2)
    f7_real_conv = tf.nn.conv2d(img, f7_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f7_imag_conv = tf.nn.conv2d(img, f7_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f7_energy = tf.sqrt(tf.square(f7_real_conv) + tf.square(f7_imag_conv))
    f8_real = np.expand_dims(gabor_bank['f8_real'], 2)
    f8_imag = np.expand_dims(gabor_bank['f8_imag'], 2)
    f8_real_conv = tf.nn.conv2d(img, f8_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f8_imag_conv = tf.nn.conv2d(img, f8_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f8_energy = tf.sqrt(tf.square(f8_real_conv) + tf.square(f8_imag_conv))
    f9_real = np.expand_dims(gabor_bank['f9_real'], 2)
    f9_imag = np.expand_dims(gabor_bank['f9_imag'], 2)
    f9_real_conv = tf.nn.conv2d(img, f9_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f9_imag_conv = tf.nn.conv2d(img, f9_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f9_energy = tf.sqrt(tf.square(f9_real_conv) + tf.square(f9_imag_conv))
    # concatenate gabor energy
    gabor_energy = tf.concat([f1_energy, f2_energy, f3_energy,
                              f4_energy, f5_energy, f6_energy,
                              f7_energy, f8_energy, f9_energy], 3)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(input_imgs.shape[2]):
            x = input_imgs[..., i].T
            x = np.expand_dims(x, 0)
            x = np.expand_dims(x, 3)
            ge = sess.run(gabor_energy, feed_dict={img: x})
            np.save('gabor_feat_%s.npy'%(i), ge)

def tfprf(input_imgs, vxl_rsp, gabor_bank):
    """multivariate-normal based pRF model."""
    # var for input data
    img = tf.placeholder("float", [None, 500, 500, 1])
    rsp_ = tf.placeholder("float", [None,])
    # config for the gabor filters
    gabor_real = np.expand_dims(gabor_bank['gabor_real'], 2)
    gabor_imag = np.expand_dims(gabor_bank['gabor_imag'], 2)
    real_conv = tf.nn.conv2d(img, gabor_real, strides=[1, 1, 1, 1],
                             padding='SAME')
    imag_conv = tf.nn.conv2d(img, gabor_imag, strides=[1, 1, 1, 1],
                             padding='SAME')
    gabor_energy = tf.sqrt(tf.square(real_conv) + tf.square(imag_conv))
    # resize features
    gabor_energy = tf.image.resize_images(gabor_energy, [250, 250])
    # reshape gabor energy for pRF masking
    gabor_energy = tf.transpose(gabor_energy, perm=[1, 2, 3, 0])
    gabor_vtr = tf.reshape(gabor_energy, [62500, -1])
    #gabor_vtr = tf.reshape(gabor_energy, [250000, -1])
    center_loc = tf.Variable(tf.multiply(tf.ones(2), 125), name='center_loc')
    sigma = tf.Variable(tf.ones(2), name='sigma')
    xinds, yinds = np.unravel_index(range(250*250), (250, 250))
    inds = (np.column_stack((xinds, yinds))).astype(np.float32)
    inds = tf.constant(inds)
    mvn = ds.MultivariateNormalDiag(loc=center_loc, scale_diag=sigma,
                                validate_args=True, allow_nan_stats=False)
    kernel = mvn.prob(inds)
    kernel = tf.reshape(kernel, (1, 62500))
    #kernel = tf.reshape(kernel, (1, 250000))
    # get features from pooling field
    vxl_feats = tf.matmul(kernel, gabor_vtr)
    vxl_feats = tf.reshape(vxl_feats, (72, -1))
    # vars for feature weights
    b = tf.Variable(tf.random_normal([1], stddev=0.001), name='b')
    w = tf.Variable(tf.random_normal([1, 72], stddev=0.001), name='W')
    vxl_wt_feats = tf.matmul(w, vxl_feats)
    rsp = vxl_wt_feats + b 

    # calculate fitting error
    error = tf.reduce_mean(tf.square(tf.reshape(rsp, [-1]) - rsp_))
    l2_error = 100*(tf.nn.l2_loss(w) + tf.nn.l2_loss(b))
    total_error = error + l2_error
    opt = tf.train.GradientDescentOptimizer(0.001)
    
    # graph config
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    vars_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    solver =  opt.minimize(total_error, var_list = vars_x)

    # model training
    batch_size = 20
    index_in_epoch = 0
    epochs_completed = 0
    for i in range(40):
        print 'Step %s'%(i)
        start = index_in_epoch
        if epochs_completed==0 and start==0:
            perm0 = np.arange(input_imgs.shape[2])
            np.random.shuffle(perm0)
            shuffle_imgs = input_imgs[..., perm0]
            shuffle_rsp = vxl_rsp[perm0]
        # go to next epoch
        if start + batch_size > input_imgs.shape[2]:
            # finish epoch
            epochs_completed += 1
            # get the rest examples in this epoch
            rest_num_examples = int(input_imgs.shape[2]) - start
            img_rest_part = shuffle_imgs[..., start:input_imgs.shape[2]]
            rsp_rest_part = shuffle_rsp[start:input_imgs.shape[2]]
            # shuffle the data
            perm = np.arange(input_imgs.shape[2])
            np.random.shuffle(perm)
            shuffle_imgs = input_imgs[..., perm]
            shuffle_rsp = vxl_rsp[perm]
            # start next epoch
            start = 0
            index_in_epoch = batch_size - rest_num_examples
            end = index_in_epoch
            img_new_part = shuffle_imgs[..., start:end]
            rsp_new_part = shuffle_rsp[start:end]
            img_batch = np.concatenate((img_rest_part, img_new_part), axis=2)
            img_batch = np.transpose(img_batch, (2, 0, 1))
            img_batch = np.expand_dims(img_batch, 3)
            batch = [img_batch,
                     np.concatenate((rsp_rest_part, rsp_new_part), axis=0)]
        else:
            index_in_epoch += batch_size
            end = index_in_epoch
            img_batch = shuffle_imgs[..., start:end]
            img_batch = np.transpose(img_batch, (2, 0, 1))
            img_batch = np.expand_dims(img_batch, 3)
            batch = [img_batch, shuffle_rsp[start:end]]
        _, step_error, step_center, step_sigma, step_b, step_w = sess.run(
                [solver, total_error, center_loc, sigma, b, w],
                                feed_dict={img: batch[0], rsp_: batch[1]})
        print 'Error: %s'%(step_error)
        print 'Center:',
        print step_center
        print 'Sigma:',
        print step_sigma
        print 'weights:',
        print step_w
        print 'bias:',
        print step_b
        np.save('prf_step%s.npy'%(i), step_prf)
    return step_center, step_sigma, step_b, step_w

def tfprf_laplacian(input_imgs, vxl_rsp, gabor_bank):
    """multivariate-normal based pRF model."""
    # var for input data
    img = tf.placeholder("float", [None, 500, 500, 1])
    rsp_ = tf.placeholder("float", [None,])
    # config for the gabor filters
    f1_real = np.expand_dims(gabor_bank['f1_real'], 2)
    f1_imag = np.expand_dims(gabor_bank['f1_imag'], 2)
    f1_real_conv = tf.nn.conv2d(img, f1_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f1_imag_conv = tf.nn.conv2d(img, f1_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f1_energy = tf.sqrt(tf.square(f1_real_conv) + tf.square(f1_imag_conv))
    f1_energy = tf.transpose(f1_energy, perm=[1, 2, 3, 0])
    f1_energy = tf.reshape(f1_energy, [62500, -1])
    f2_real = np.expand_dims(gabor_bank['f2_real'], 2)
    f2_imag = np.expand_dims(gabor_bank['f2_imag'], 2)
    f2_real_conv = tf.nn.conv2d(img, f2_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f2_imag_conv = tf.nn.conv2d(img, f2_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f2_energy = tf.sqrt(tf.square(f2_real_conv) + tf.square(f2_imag_conv))
    f2_energy = tf.transpose(f2_energy, perm=[1, 2, 3, 0])
    f2_energy = tf.reshape(f2_energy, [62500, -1])
    f3_real = np.expand_dims(gabor_bank['f3_real'], 2)
    f3_imag = np.expand_dims(gabor_bank['f3_imag'], 2)
    f3_real_conv = tf.nn.conv2d(img, f3_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f3_imag_conv = tf.nn.conv2d(img, f3_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f3_energy = tf.sqrt(tf.square(f3_real_conv) + tf.square(f3_imag_conv))
    f3_energy = tf.transpose(f3_energy, perm=[1, 2, 3, 0])
    f3_energy = tf.reshape(f3_energy, [62500, -1])
    f4_real = np.expand_dims(gabor_bank['f4_real'], 2)
    f4_imag = np.expand_dims(gabor_bank['f4_imag'], 2)
    f4_real_conv = tf.nn.conv2d(img, f4_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f4_imag_conv = tf.nn.conv2d(img, f4_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f4_energy = tf.sqrt(tf.square(f4_real_conv) + tf.square(f4_imag_conv))
    f4_energy = tf.transpose(f4_energy, perm=[1, 2, 3, 0])
    f4_energy = tf.reshape(f4_energy, [62500, -1])
    f5_real = np.expand_dims(gabor_bank['f5_real'], 2)
    f5_imag = np.expand_dims(gabor_bank['f5_imag'], 2)
    f5_real_conv = tf.nn.conv2d(img, f5_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f5_imag_conv = tf.nn.conv2d(img, f5_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f5_energy = tf.sqrt(tf.square(f5_real_conv) + tf.square(f5_imag_conv))
    f5_energy = tf.transpose(f5_energy, perm=[1, 2, 3, 0])
    f5_energy = tf.reshape(f5_energy, [62500, -1])
    f6_real = np.expand_dims(gabor_bank['f6_real'], 2)
    f6_imag = np.expand_dims(gabor_bank['f6_imag'], 2)
    f6_real_conv = tf.nn.conv2d(img, f6_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f6_imag_conv = tf.nn.conv2d(img, f6_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f6_energy = tf.sqrt(tf.square(f6_real_conv) + tf.square(f6_imag_conv))
    f6_energy = tf.transpose(f6_energy, perm=[1, 2, 3, 0])
    f6_energy = tf.reshape(f6_energy, [62500, -1])
    f7_real = np.expand_dims(gabor_bank['f7_real'], 2)
    f7_imag = np.expand_dims(gabor_bank['f7_imag'], 2)
    f7_real_conv = tf.nn.conv2d(img, f7_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f7_imag_conv = tf.nn.conv2d(img, f7_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f7_energy = tf.sqrt(tf.square(f7_real_conv) + tf.square(f7_imag_conv))
    f7_energy = tf.transpose(f7_energy, perm=[1, 2, 3, 0])
    f7_energy = tf.reshape(f7_energy, [62500, -1])
    f8_real = np.expand_dims(gabor_bank['f8_real'], 2)
    f8_imag = np.expand_dims(gabor_bank['f8_imag'], 2)
    f8_real_conv = tf.nn.conv2d(img, f8_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f8_imag_conv = tf.nn.conv2d(img, f8_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f8_energy = tf.sqrt(tf.square(f8_real_conv) + tf.square(f8_imag_conv))
    f8_energy = tf.transpose(f8_energy, perm=[1, 2, 3, 0])
    f8_energy = tf.reshape(f8_energy, [62500, -1])
    f9_real = np.expand_dims(gabor_bank['f9_real'], 2)
    f9_imag = np.expand_dims(gabor_bank['f9_imag'], 2)
    f9_real_conv = tf.nn.conv2d(img, f9_real, strides=[1, 2, 2, 1],
                                padding='SAME')
    f9_imag_conv = tf.nn.conv2d(img, f9_imag, strides=[1, 2, 2, 1],
                                padding='SAME')
    f9_energy = tf.sqrt(tf.square(f9_real_conv) + tf.square(f9_imag_conv))
    f9_energy = tf.transpose(f9_energy, perm=[1, 2, 3, 0])
    f9_energy = tf.reshape(f9_energy, [62500, -1])
    # var for feature pooling field
    fpf_kernel = tf.random_normal([1, 250, 250, 1], stddev=0.001)
    blur_kernel = np.array([[1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256],
                            [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                            [6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256],
                            [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                            [1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256]])
    blur_kernel = np.expand_dims(blur_kernel, 2)
    blur_kernel = np.expand_dims(blur_kernel, 3)
    fpf_kernel = tf.nn.conv2d(fpf_kernel, blur_kernel, strides=[1, 1, 1, 1],
                              padding='SAME')
    fpf_kernel = tf.nn.conv2d(fpf_kernel, blur_kernel, strides=[1, 1, 1, 1],
                              padding='SAME')
    fpf = tf.Variable(tf.reshape(fpf_kernel, [250, 250]), name='fpf')
    flat_fpf = tf.reshape(fpf, (1, 62500))
    # get features from pooling field
    f1_feats = tf.matmul(flat_fpf, f1_energy)
    f1_feats = tf.reshape(f1_feats, (8, -1))
    f2_feats = tf.matmul(flat_fpf, f2_energy)
    f2_feats = tf.reshape(f2_feats, (8, -1))
    f3_feats = tf.matmul(flat_fpf, f3_energy)
    f3_feats = tf.reshape(f3_feats, (8, -1))
    f4_feats = tf.matmul(flat_fpf, f4_energy)
    f4_feats = tf.reshape(f4_feats, (8, -1))
    f5_feats = tf.matmul(flat_fpf, f5_energy)
    f5_feats = tf.reshape(f5_feats, (8, -1))
    f6_feats = tf.matmul(flat_fpf, f6_energy)
    f6_feats = tf.reshape(f6_feats, (8, -1))
    f7_feats = tf.matmul(flat_fpf, f7_energy)
    f7_feats = tf.reshape(f7_feats, (8, -1))
    f8_feats = tf.matmul(flat_fpf, f8_energy)
    f8_feats = tf.reshape(f8_feats, (8, -1))
    f9_feats = tf.matmul(flat_fpf, f9_energy)
    f9_feats = tf.reshape(f9_feats, (8, -1))
    # concatenate gabor features within fpf
    vxl_feats = tf.concat([f1_feats, f2_feats, f3_feats,
                           f4_feats, f5_feats, f6_feats,
                           f7_feats, f8_feats, f9_feats], 0)
    # vars for feature weights
    b = tf.Variable(tf.random_normal([1], stddev=0.001), name='b')
    w = tf.Variable(tf.constant(0.001, shape=[1, 72]), name='W')
    vxl_wt_feats = tf.matmul(w, vxl_feats)
    rsp = vxl_wt_feats + b 

    # calculate fitting error
    error = tf.reduce_mean(tf.square(tf.reshape(rsp, [-1]) - rsp_))
    # parameter regularization
    l2_error = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    # laplacian regularization
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_kernel = np.expand_dims(laplacian_kernel, 2)
    laplacian_kernel = np.expand_dims(laplacian_kernel, 3)
    fpf_shadow = tf.expand_dims(tf.expand_dims(fpf, 0), 3)
    laplacian_reg = tf.nn.conv2d(fpf_shadow, laplacian_kernel,
                                 strides=[1, 1, 1, 1], padding='VALID')
    reg_error = tf.reduce_sum(tf.square(laplacian_reg))
    # get total error
    total_error = 10*error + l2_error + reg_error
    opt = tf.train.GradientDescentOptimizer(0.005)
    
    # graph config
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    vars_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    solver =  opt.minimize(total_error, var_list = vars_x)

    # model training
    batch_size = 10
    index_in_epoch = 0
    epochs_completed = 0
    for i in range(1001):
        print 'Step %s'%(i)
        start = index_in_epoch
        if epochs_completed==0 and start==0:
            perm0 = np.arange(input_imgs.shape[2])
            np.random.shuffle(perm0)
            shuffle_imgs = input_imgs[..., perm0]
            shuffle_rsp = vxl_rsp[perm0]
        # go to next epoch
        if start + batch_size > input_imgs.shape[2]:
            # finish epoch
            epochs_completed += 1
            # get the rest examples in this epoch
            rest_num_examples = int(input_imgs.shape[2]) - start
            img_rest_part = shuffle_imgs[..., start:input_imgs.shape[2]]
            rsp_rest_part = shuffle_rsp[start:input_imgs.shape[2]]
            # shuffle the data
            perm = np.arange(input_imgs.shape[2])
            np.random.shuffle(perm)
            shuffle_imgs = input_imgs[..., perm]
            shuffle_rsp = vxl_rsp[perm]
            # start next epoch
            start = 0
            index_in_epoch = batch_size - rest_num_examples
            end = index_in_epoch
            img_new_part = shuffle_imgs[..., start:end]
            rsp_new_part = shuffle_rsp[start:end]
            img_batch = np.concatenate((img_rest_part, img_new_part), axis=2)
            img_batch = np.transpose(img_batch, (2, 0, 1))
            img_batch = np.expand_dims(img_batch, 3)
            batch = [img_batch,
                     np.concatenate((rsp_rest_part, rsp_new_part), axis=0)]
        else:
            index_in_epoch += batch_size
            end = index_in_epoch
            img_batch = shuffle_imgs[..., start:end]
            img_batch = np.transpose(img_batch, (2, 0, 1))
            img_batch = np.expand_dims(img_batch, 3)
            batch = [img_batch, shuffle_rsp[start:end]]
        _, step_error, step_fpf, step_b, step_w = sess.run(
                [solver, total_error, fpf, b, w],
                                feed_dict={img: batch[0], rsp_: batch[1]})
        print 'Error: %s'%(step_error)
        print 'weights:',
        print step_w
        print 'bias:',
        print step_b
        if i%10==0:
            fig, ax = plt.subplots()
            cax = ax.imshow(step_fpf, cmap='gray')
            fig.colorbar(cax)
            plt.savefig('fpf_step%s.png'%(i))
            plt.close(fig)
    return step_b, step_w

def tfprf_laplacian_tmp(input_imgs, vxl_rsp):
    """multivariate-normal based pRF model."""
    # var for input data
    img = tf.placeholder("float", [None, 500, 500, 72])
    rsp_ = tf.placeholder("float", [None,])
    # reshape gabor energy for fpf masking
    gabor_energy = tf.transpose(img, perm=[1, 2, 3, 0])
    gabor_vtr = tf.reshape(gabor_energy, [250000, -1])
    # var for feature pooling field
    fpf = tf.Variable(tf.random_normal([500, 500], stddev=0.001), name='fpf')
    flat_fpf = tf.reshape(fpf, (1, 250000))
    # get features from pooling field
    vxl_feats = tf.matmul(flat_fpf, gabor_vtr)
    vxl_feats = tf.reshape(vxl_feats, (72, -1))
    # vars for feature weights
    b = tf.Variable(tf.random_normal([1], stddev=0.001), name='b')
    w = tf.Variable(tf.random_normal([1, 72], stddev=0.001), name='W')
    vxl_wt_feats = tf.matmul(w, vxl_feats)
    rsp = vxl_wt_feats + b 

    # calculate fitting error
    error = tf.reduce_mean(tf.square(tf.reshape(rsp, [-1]) - rsp_))
    # parameter regularization
    l2_error = 10*(tf.nn.l2_loss(w) + tf.nn.l2_loss(b))
    # laplacian regularization
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    laplacian_kernel = np.expand_dims(laplacian_kernel, 2)
    laplacian_kernel = np.expand_dims(laplacian_kernel, 3)
    fpf_shadow = tf.expand_dims(tf.expand_dims(fpf, 0), 3)
    laplacian_reg = tf.nn.conv2d(fpf_shadow, laplacian_kernel,
                                 strides=[1, 1, 1, 1], padding='VALID')
    reg_error = tf.reduce_sum(tf.square(laplacian_reg))
    # get total error
    total_error = error + l2_error + reg_error
    opt = tf.train.GradientDescentOptimizer(0.005)
    
    # graph config
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.95
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    vars_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    solver =  opt.minimize(total_error, var_list = vars_x)

    # model training
    batch_size = 10
    index_in_epoch = 0
    epochs_completed = 0
    for i in range(1000):
        print 'Step %s'%(i)
        start = index_in_epoch
        if epochs_completed==0 and start==0:
            perm = np.arange(input_imgs.shape[0])
            np.random.shuffle(perm)
        # go to next epoch
        if start + batch_size > input_imgs.shape[0]:
            # finish epoch
            epochs_completed += 1
            # get the rest examples in this epoch
            rest_num_examples = int(input_imgs.shape[0]) - start
            img_rest_part = input_imgs[perm[start:input_imgs.shape[0]]]
            rsp_rest_part = vxl_rsp[perm[start:input_imgs.shape[0]]]
            # shuffle the data
            perm = np.arange(input_imgs.shape[0])
            np.random.shuffle(perm)
            # start next epoch
            start = 0
            index_in_epoch = batch_size - rest_num_examples
            end = index_in_epoch
            img_new_part = input_imgs[perm[start:end]]
            rsp_new_part = vxl_rsp[perm[start:end]]
            img_batch = np.concatenate((img_rest_part, img_new_part), axis=0)
            batch = [img_batch,
                     np.concatenate((rsp_rest_part, rsp_new_part), axis=0)]
        else:
            index_in_epoch += batch_size
            end = index_in_epoch
            batch = [input_imgs[perm[start:end]], vxl_rsp[perm[start:end]]]
        _, step_error, step_fpf, step_b, step_w = sess.run(
                [solver, total_error, fpf, b, w],
                                feed_dict={img: batch[0], rsp_: batch[1]})
        print 'Error: %s'%(step_error)
        print 'weights:',
        print step_w
        print 'bias:',
        print step_b
        if i%20==0:
            fig, ax = plt.subplots()
            cax = ax.imshow(step_fpf, cmap='gray')
            fig.colorbar(cax)
            plt.savefig('fpf_step%s.png'%(i))
        #np.save('fpf_step%s.npy'%(i), step_fpf)
    return step_b, step_w
    #return step_center, step_sigma, step_b, step_w


if __name__ == '__main__':
    """Main function"""
    # database directory config
    # config parser
    cf = configParser.Config('config')
    # database directory config
    db_dir = os.path.join(cf.get('database', 'path'), 'vim1')
    # directory config for analysis
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures', 'vim1')
    res_dir = os.path.join(root_dir, 'subjects')
    
    ## directory config for analysis
    #root_dir = r'/nfs/home/cddu/ActMax'
    #feat_dir = root_dir
    #db_dir = os.path.join(root_dir, 'db')
    #res_dir = os.path.join(root_dir, 'subjects')

    #-- general config
    subj_id = 1
    roi = 'v1'
    # directory config
    subj_dir = os.path.join(res_dir, 'vim1_S%s'%(subj_id))
    prf_dir = os.path.join(subj_dir, 'prf')

    #-- parameter preparation
    #gabor_bank_file = os.path.join(feat_dir, 'gabor_kernels.npz')
    gabor_bank_file = os.path.join(feat_dir, 'gabor_kernels_small.npz')
    gabor_bank = np.load(gabor_bank_file)
    #vxl_coding_paras_file =os.path.join(prf_dir,'tfrecon','vxl_coding_wts.npz')
    #vxl_coding_paras = np.load(vxl_coding_paras_file)

    #-- test encoding model
    #print 'Select voxel index',
    #print vxl_coding_paras['vxl_idx']
    #img_file = os.path.join(root_dir, 'example_imgs.npy')
    #imgs = np.load(img_file)
    #model_test(imgs, gabor_bank, vxl_coding_paras)

    #-- test gabor banks
    #gabor_test(imgs, gabor_bank)

    #-- stimuli reconstruction
    #resp_file = os.path.join(db_dir, 'EstimatedResponses.mat')
    #resp_mat = tables.open_file(resp_file)
    ## create mask
    ## train data shape: (1750, ~25000)
    #train_ts = resp_mat.get_node('/dataTrnS%s'%(subj_id))[:]
    ## reshape fmri response: data shape (#voxel, 1750/120)
    #train_ts = np.nan_to_num(train_ts.T)
    #m = np.mean(train_ts, axis=1, keepdims=True)
    #s = np.std(train_ts, axis=1, keepdims=True)
    #train_ts = (train_ts - m) / (s + 1e-5)
    ##val_ts = tf.get_node('/dataValS%s'%(subj_id))[:]
    ##val_ts = val_ts.T
    ##val_ts = np.nan_to_num(val_ts[vxl_idx])
    #resp_mat.close()
    #y_ = train_ts[vxl_coding_paras['vxl_idx'].astype(np.int)]
    ## shape: (#voxel, 1750)
    #print y_.shape
    #recon_img = reconstructor(gabor_bank, vxl_coding_paras, y_)
    ## show image    
    #fig=plt.figure()
    #plt.imshow(recon_img. cmap='gray')
    #plt.savefig('recons.png')

    # multivariate normal based pRF
    stimuli_file = os.path.join(db_dir, 'stimuli', 'train_stimuli.npy')
    input_imgs = np.load(stimuli_file)
    img_m = np.mean(input_imgs, axis=2, keepdims=True)
    input_imgs = input_imgs - img_m
    #stimuli_file = os.path.join(feat_dir, 'train_stimuli_gabor_feat.memdat')
    #input_imgs = np.memmap(stimuli_file, dtype='float32', mode='r',
    #                       shape=(1750, 500, 500, 72))
    resp_file = os.path.join(db_dir, 'EstimatedResponses.mat')
    resp_mat = tables.open_file(resp_file)
    train_ts = resp_mat.get_node('/dataTrnS%s'%(subj_id))[:]
    train_ts = np.nan_to_num(train_ts.T)
    resp_mat.close()
    ts_m = np.mean(train_ts, axis=1, keepdims=True)
    ts_s = np.std(train_ts, axis=1, keepdims=True)
    train_ts = (train_ts - ts_m) / (ts_s + 1e-5)
    # select voxel 19165 as an example
    vxl_rsp = train_ts[24031]
    print 'Image data shape ',
    print input_imgs.shape
    print 'Voxel time point number',
    print vxl_rsp.shape
    #tfprf_laplacian_tmp(input_imgs, vxl_rsp)
    tfprf_laplacian(input_imgs, vxl_rsp, gabor_bank)
    #tfprf(input_imgs, vxl_rsp, gabor_bank)
    #np.save('prf_example.npy', prf)

