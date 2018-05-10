# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os    
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
from math import log
from collections import Counter
from scipy.misc import imresize
import tables
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from braincode.util import configParser
from braincode.prf import dataio

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

def variable_summaries(var):
    """Attach a lot of summaries to Tensor for TensorBoard visualization."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def tfprf_laplacian(input_imgs, vxl_rsp, gabor_bank, vxl_dir):
    """laplacian regularized pRF model."""
    # get image mask
    img_m = np.mean(input_imgs, axis=2)
    img_mask = imresize(img_m, (250, 250))
    # resized image value range: 0-255
    img_mask = np.reshape(img_mask<170, [-1])

    graph = tf.Graph()
    with graph.as_default():
        # vars for input data
        with tf.name_scope('input'):
            img = tf.placeholder("float", [None, 500, 500, 1], name='input-img')
            rsp_ = tf.placeholder("float", [None,], name='vxl-rsp')

        # var for feature pooling field
        with tf.name_scope('pooling-field'):
            fpf_kernel = tf.random_normal([1, 250, 250, 1], stddev=0.01)
            blur = np.array([[1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256],
                             [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                             [6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256],
                             [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                             [1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256]])
            blur = np.expand_dims(blur, 2)
            blur = np.expand_dims(blur, 3)
            fpf_kernel = tf.nn.conv2d(fpf_kernel, blur, strides=[1, 1, 1, 1],
                                      padding='SAME')
            #fpf_kernel = tf.nn.conv2d(fpf_kernel, blur, strides=[1, 1, 1, 1],
            #                          padding='SAME')
            fpf = tf.Variable(tf.reshape(fpf_kernel, [250, 250]), name='fpf')
            flat_fpf = tf.transpose(tf.boolean_mask(tf.reshape(tf.nn.relu(fpf),
                                                               (62500, 1)),
                                                    img_mask), [1, 0])

        # gabor features extraction
        with tf.name_scope('feature-extract'):
            feat_vtr = []
            for i in range(9):
                # config for the gabor filters
                gabor_real = np.expand_dims(gabor_bank['f%s_real'%(i+1)], 2)
                gabor_imag = np.expand_dims(gabor_bank['f%s_imag'%(i+1)], 2)
                rconv = tf.nn.conv2d(img, gabor_real, strides=[1, 2, 2, 1],
                                     padding='SAME')
                iconv = tf.nn.conv2d(img, gabor_imag, strides=[1, 2, 2, 1],
                                     padding='SAME')
                gabor_energy = tf.sqrt(tf.square(rconv) + tf.square(iconv))
                gabor_energy = tf.transpose(gabor_energy, perm=[1, 2, 3, 0])
                gabor_energy = tf.boolean_mask(tf.reshape(gabor_energy,
                                                        [62500, -1]), img_mask)
                # get feature summary from pooling field
                gabor_feat = tf.reshape(tf.matmul(flat_fpf, gabor_energy),
                                        (8, -1))
                feat_vtr.append(gabor_feat)
            # concatenate gabor features within fpf
            vxl_feats = tf.concat(feat_vtr, 0)

        # vars for feature weights
        with tf.name_scope('weighted-features'):
            b = tf.Variable(tf.constant(0.01, shape=[1]), name='bias')
            variable_summaries(b)
            w = tf.Variable(tf.constant(0.01, shape=[1, 72]), name='weights')
            variable_summaries(w)
            vxl_wt_feats = tf.matmul(w, vxl_feats)
            rsp = tf.reshape(vxl_wt_feats + b, [-1])

        # loss defination
        with tf.name_scope('loss'):
            # calculate fitting error
            error = tf.reduce_mean(tf.square(rsp - rsp_))
            # parameter regularization
            #l2_error = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
            # laplacian regularization
            laplacian_kernel = np.array([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]])
            laplacian_kernel = np.expand_dims(laplacian_kernel, 2)
            laplacian_kernel = np.expand_dims(laplacian_kernel, 3)
            fpf_shadow = tf.expand_dims(tf.expand_dims(fpf, 0), 3)
            laplacian_error = tf.reduce_sum(tf.square(tf.nn.conv2d(fpf_shadow,
                                                         laplacian_kernel,
                                                         strides=[1, 1, 1, 1],
                                                         padding='VALID')))
            #l1_error = tf.reduce_sum(tf.abs(fpf))
            # get total error
            total_error = 10*error + 10*laplacian_error

        tf.summary.scalar('fitting-loss', error)
        tf.summary.scalar('total-loss', total_error)

        # for model saving
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        vars_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        solver =  tf.train.AdamOptimizer(0.0005).minimize(total_error,
                                                         var_list = vars_x)
        # merge summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(vxl_dir, 'train'),
                                             sess.graph)
        #test_writer = tf.summary.FileWriter('./test')
        sess.run(tf.global_variables_initializer())

        # data splitting
        input_imgs = input_imgs - np.expand_dims(img_m, 2)
        sample_num = input_imgs.shape[2]
        train_imgs = input_imgs[..., :int(sample_num*0.9)]
        val_imgs = input_imgs[..., int(sample_num*0.9):]
        val_imgs = np.transpose(val_imgs, (2, 0, 1))
        val_imgs = np.expand_dims(val_imgs, 3)
        train_rsp = vxl_rsp[:int(sample_num*0.9)]
        val_rsp = vxl_rsp[int(sample_num*0.9):]
        #print train_imgs.shape
        #print val_imgs.shape
        #print train_rsp.shape
        #print val_rsp.shape

        # model training
        batch_size = 9
        index_in_epoch = 0
        epochs_completed = 0
        min_err = None
        patience_cnt = 0
        patience = 6
        iter_num = 0
        val_loss = []
        while 1:
            start = index_in_epoch
            if epochs_completed==0 and start==0:
                perm0 = np.arange(train_imgs.shape[2])
                np.random.shuffle(perm0)
                shuffle_imgs = train_imgs[..., perm0]
                shuffle_rsp = train_rsp[perm0]
            # go to next epoch
            if start + batch_size > train_imgs.shape[2]:
                # finish epoch
                epochs_completed += 1
                # get the rest examples in this epoch
                rest_num_examples = int(train_imgs.shape[2]) - start
                img_rest_part = shuffle_imgs[..., start:train_imgs.shape[2]]
                rsp_rest_part = shuffle_rsp[start:train_imgs.shape[2]]
                # shuffle the data
                perm = np.arange(train_imgs.shape[2])
                np.random.shuffle(perm)
                shuffle_imgs = train_imgs[..., perm]
                shuffle_rsp = train_rsp[perm]
                # start next epoch
                start = 0
                index_in_epoch = batch_size - rest_num_examples
                end = index_in_epoch
                img_new_part = shuffle_imgs[..., start:end]
                rsp_new_part = shuffle_rsp[start:end]
                img_batch = np.concatenate((img_rest_part,img_new_part), axis=2)
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
            _, summary, step_error, step_fpf = sess.run(
                                    [solver, merged, total_error, fpf],
                                    feed_dict={img: batch[0], rsp_: batch[1]})
            train_writer.add_summary(summary, iter_num)
            if (iter_num+1)%175==0:
                print 'Ep %s'%((iter_num+1)/175)
                print 'Training Error: %s'%(step_error)
                rsp_err = sess.run(error, feed_dict={img: batch[0],
                                                     rsp_: batch[1]})
                #l2_err = sess.run(l2_error, feed_dict={img:batch[0],
                #                                       rsp_: batch[1]})
                lap_err = sess.run(laplacian_error, feed_dict={img:batch[0],
                                                               rsp_: batch[1]})
                #l1_err = sess.run(l1_error, feed_dict={img:batch[0],
                #                                       rsp_: batch[1]})
                print 'Rsp error: %s'%(rsp_err)
                #print 'L2 error: %s'%(l2_err)
                print 'Laplacian error: %s'%(lap_err)
                #print 'L1 error: %s'%(l1_err)
                # model validation
                pred_val_rsp = np.zeros(175)
                for j in range(35):
                    part_rsp = sess.run(rsp,
                                feed_dict={img: val_imgs[(j*5):(j*5+5)],
                                           rsp_: val_rsp[(j*5):(j*5+5)]})
                    pred_val_rsp[(j*5):(j*5+5)] = part_rsp
                val_err = np.mean(np.square(pred_val_rsp - val_rsp))
                print 'Validation Error: %s'%(val_err)
                val_loss.append(val_err)
                #val_corr = np.corrcoef(pred_val_rsp, val_rsp)[0, 1]
                #print 'Validation Corr: %s'%(val_corr)
                if iter_num==174:
                    min_err = val_err
                else:
                    if (min_err - val_err) >= 0.0025:
                        min_err = val_err
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                # stop signal
                if patience_cnt > patience:
                    print 'Early stopping - step %s'%(iter_num)
                    # plot fpf
                    fig, ax = plt.subplots()
                    cax = ax.imshow(step_fpf, cmap='gray')
                    fig.colorbar(cax)
                    plt.savefig(os.path.join(vxl_dir,
                            'fpf_epoch%s.png'%((iter_num+1)/175)))
                    plt.close(fig)
                    # save model
                    saver.save(sess, os.path.join(vxl_dir, 'prf_model'),
                               global_step=(iter_num - (patience+1)*175))
                    saver.save(sess, os.path.join(vxl_dir, 'prf_model'),
                               global_step=iter_num, write_meta_graph=False)
                    # save final validation loss
                    with open(os.path.join(vxl_dir, 'val_loss.txt'), 'w+') as f:
                        val_idx = -1 * patience - 2
                        f.write('%s\n'%(val_loss[val_idx]))
                    break
            iter_num += 1

        train_writer.close()
        #test_writer.close()
    return

def tfprf_test(train_imgs, val_imgs, vxl_rsp, gabor_bank, vxl_dir):
    """Test laplacian regularized pRF model on test dataset."""
    # get image mask
    img_m = np.mean(train_imgs, axis=2)
    img_mask = imresize(img_m, (250, 250))
    # resized image value range: 0-255
    img_mask = np.reshape(img_mask<170, [-1])

    graph = tf.Graph()
    with graph.as_default():
        # vars for input data
        with tf.name_scope('input'):
            img = tf.placeholder("float", [None, 500, 500, 1], name='input-img')
            rsp_ = tf.placeholder("float", [None,], name='vxl-rsp')

        # var for feature pooling field
        with tf.name_scope('pooling-field'):
            fpf_kernel = tf.random_normal([1, 250, 250, 1], stddev=0.01)
            blur = np.array([[1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256],
                             [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                             [6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256],
                             [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                             [1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256]])
            blur = np.expand_dims(blur, 2)
            blur = np.expand_dims(blur, 3)
            fpf_kernel = tf.nn.conv2d(fpf_kernel, blur, strides=[1, 1, 1, 1],
                                      padding='SAME')
            #fpf_kernel = tf.nn.conv2d(fpf_kernel, blur, strides=[1, 1, 1, 1],
            #                          padding='SAME')
            fpf = tf.Variable(tf.reshape(fpf_kernel, [250, 250]), name='fpf')
            flat_fpf = tf.transpose(tf.boolean_mask(tf.reshape(tf.nn.relu(fpf),
                                                               (62500, 1)),
                                                    img_mask), [1, 0])

        # gabor features extraction
        with tf.name_scope('feature-extract'):
            feat_vtr = []
            for i in range(9):
                # config for the gabor filters
                gabor_real = np.expand_dims(gabor_bank['f%s_real'%(i+1)], 2)
                gabor_imag = np.expand_dims(gabor_bank['f%s_imag'%(i+1)], 2)
                rconv = tf.nn.conv2d(img, gabor_real, strides=[1, 2, 2, 1],
                                     padding='SAME')
                iconv = tf.nn.conv2d(img, gabor_imag, strides=[1, 2, 2, 1],
                                     padding='SAME')
                gabor_energy = tf.sqrt(tf.square(rconv) + tf.square(iconv))
                gabor_energy = tf.transpose(gabor_energy, perm=[1, 2, 3, 0])
                gabor_energy = tf.boolean_mask(tf.reshape(gabor_energy,
                                                        [62500, -1]), img_mask)
                # get feature summary from pooling field
                gabor_feat = tf.reshape(tf.matmul(flat_fpf, gabor_energy),
                                        (8, -1))
                feat_vtr.append(gabor_feat)
            # concatenate gabor features within fpf
            vxl_feats = tf.concat(feat_vtr, 0)

        # vars for feature weights
        with tf.name_scope('weighted-features'):
            b = tf.Variable(tf.constant(0.01, shape=[1]), name='bias')
            w = tf.Variable(tf.constant(0.01, shape=[1, 72]), name='weights')
            vxl_wt_feats = tf.matmul(w, vxl_feats)
            rsp = tf.reshape(vxl_wt_feats + b, [-1])

        # loss defination
        with tf.name_scope('loss'):
            # calculate fitting error
            error = tf.reduce_mean(tf.square(rsp - rsp_))
            # parameter regularization
            # laplacian regularization
            laplacian_kernel = np.array([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]])
            laplacian_kernel = np.expand_dims(laplacian_kernel, 2)
            laplacian_kernel = np.expand_dims(laplacian_kernel, 3)
            fpf_shadow = tf.expand_dims(tf.expand_dims(fpf, 0), 3)
            laplacian_error = tf.reduce_sum(tf.square(tf.nn.conv2d(fpf_shadow,
                                                         laplacian_kernel,
                                                         strides=[1, 1, 1, 1],
                                                         padding='VALID')))
            # get total error
            total_error = 10*error + 10*laplacian_error

    with tf.Session(graph=graph) as sess:
        # find the optimal model
        file_list = os.listdir(vxl_dir)
        file_list = [item for item in file_list if item[-5:]=='index']
        iter_num = [int(item.split('.')[0].split('-')[1]) for item in file_list]
        sel_iter_num = min(iter_num)
        model_path = os.path.join(vxl_dir, 'prf_model-%s'%(sel_iter_num))
        # load saved model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        # test on validation dataset
        input_imgs = val_imgs - np.expand_dims(img_m, 2)
        input_imgs = np.transpose(input_imgs, (2, 0, 1))
        input_imgs = np.expand_dims(input_imgs, 3)

        pred_val_rsp = np.zeros(120)
        for i in range(24):
            part_rsp = sess.run(rsp, feed_dict={img: input_imgs[(i*5):(i*5+5)],
                                                rsp_: vxl_rsp[(i*5):(i*5+5)]})
            pred_val_rsp[(i*5):(i*5+5)] = part_rsp
        val_err = np.mean(np.square(pred_val_rsp - vxl_rsp))
        print 'Validation Error: %s'%(val_err)
        # save final validation loss
        with open(os.path.join(vxl_dir, 'test_loss.txt'), 'w+') as f:
            f.write('%s\n'%(val_err))
    return

def tfprf_laplacian_entropy(input_imgs, vxl_rsp, gabor_bank, vxl_dir):
    """laplacian regularized pRF model."""
    # get image mask
    img_m = np.mean(input_imgs, axis=2)
    img_mask = imresize(img_m, (250, 250))
    # resized image value range: 0-255
    img_mask = np.reshape(img_mask<170, [-1])

    graph = tf.Graph()
    with graph.as_default():
        # vars for input data
        with tf.name_scope('input'):
            img = tf.placeholder("float", [None, 500, 500, 1], name='input-img')
            rsp_ = tf.placeholder("float", [None,], name='vxl-rsp')

        # var for feature pooling field
        with tf.name_scope('pooling-field'):
            fpf_kernel = tf.random_normal([1, 250, 250, 1], stddev=0.01)
            blur = np.array([[1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256],
                             [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                             [6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256],
                             [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                             [1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256]])
            blur = np.expand_dims(blur, 2)
            blur = np.expand_dims(blur, 3)
            fpf_kernel = tf.nn.conv2d(fpf_kernel, blur, strides=[1, 1, 1, 1],
                                      padding='SAME')
            fpf = tf.Variable(tf.reshape(fpf_kernel, [250, 250]), name='fpf')
            flat_fpf = tf.transpose(tf.boolean_mask(tf.reshape(fpf, (62500, 1)),
                                                    img_mask), [1, 0])

        # gabor features extraction
        with tf.name_scope('feature-extract'):
            feat_vtr = []
            for i in range(9):
                # config for the gabor filters
                gabor_real = np.expand_dims(gabor_bank['f%s_real'%(i+1)], 2)
                gabor_imag = np.expand_dims(gabor_bank['f%s_imag'%(i+1)], 2)
                rconv = tf.nn.conv2d(img, gabor_real, strides=[1, 2, 2, 1],
                                     padding='SAME')
                iconv = tf.nn.conv2d(img, gabor_imag, strides=[1, 2, 2, 1],
                                     padding='SAME')
                gabor_energy = tf.sqrt(tf.square(rconv) + tf.square(iconv))
                gabor_energy = tf.transpose(gabor_energy, perm=[1, 2, 3, 0])
                gabor_energy = tf.boolean_mask(tf.reshape(gabor_energy,
                                                        [62500, -1]), img_mask)
                # get feature summary from pooling field
                gabor_feat = tf.reshape(tf.matmul(flat_fpf, gabor_energy),
                                        (8, -1))
                feat_vtr.append(gabor_feat)
            # concatenate gabor features within fpf
            vxl_feats = tf.concat(feat_vtr, 0)

        # vars for feature weights
        with tf.name_scope('weighted-features'):
            b = tf.Variable(tf.constant(0.01, shape=[1]), name='bias')
            variable_summaries(b)
            w = tf.Variable(tf.constant(0.01, shape=[1, 72]), name='weights')
            variable_summaries(w)
            vxl_wt_feats = tf.matmul(w, vxl_feats)
            rsp = tf.reshape(vxl_wt_feats + b, [-1])

        # loss defination
        with tf.name_scope('loss'):
            # calculate fitting error
            error = tf.reduce_mean(tf.square(rsp - rsp_))
            # laplacian regularization
            laplacian_kernel = np.array([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]])
            laplacian_kernel = np.expand_dims(laplacian_kernel, 2)
            laplacian_kernel = np.expand_dims(laplacian_kernel, 3)
            fpf_shadow = tf.expand_dims(tf.expand_dims(fpf, 0), 3)
            laplacian_error = tf.reduce_sum(tf.square(tf.nn.conv2d(fpf_shadow,
                                                         laplacian_kernel,
                                                         strides=[1, 1, 1, 1],
                                                         padding='VALID')))
            # entropy regularization
            abstract_fpf = tf.nn.avg_pool(tf.sign(tf.abs(fpf_shadow)),
                                          ksize=[1, 2, 2, 1],
                                          strides=[1, 1, 1, 1],
                                          padding='VALID')
            abstract_fpf = abstract[0, ..., 0]
            counts = dict(Counter(abstract.flatten())).values()
            total_count = sum(counts)
            discrete_dist = [float(x) / total_count for x in counts]
            entropy = 0
            for item in discrete_dist:
                entropy += item * log(item, 2)
            if entropy != 0:
                entropy *= -1

            # get total error
            total_error = 10*error + laplacian_error + entropy

        tf.summary.scalar('fitting-loss', error)
        tf.summary.scalar('total-loss', total_error)

        # for model saving
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        vars_x = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        solver =  tf.train.AdamOptimizer(0.0005).minimize(total_error,
                                                         var_list = vars_x)
        # merge summaries
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(vxl_dir, 'train'),
                                             sess.graph)
        #test_writer = tf.summary.FileWriter('./test')
        sess.run(tf.global_variables_initializer())

        # data splitting
        input_imgs = input_imgs - np.expand_dims(img_m, 2)
        sample_num = input_imgs.shape[2]
        train_imgs = input_imgs[..., :int(sample_num*0.9)]
        val_imgs = input_imgs[..., int(sample_num*0.9):]
        val_imgs = np.transpose(val_imgs, (2, 0, 1))
        val_imgs = np.expand_dims(val_imgs, 3)
        train_rsp = vxl_rsp[:int(sample_num*0.9)]
        val_rsp = vxl_rsp[int(sample_num*0.9):]
        #print train_imgs.shape
        #print val_imgs.shape
        #print train_rsp.shape
        #print val_rsp.shape

        # model training
        batch_size = 9
        index_in_epoch = 0
        epochs_completed = 0
        min_err = None
        patience_cnt = 0
        patience = 6
        iter_num = 0
        val_loss = []
        while 1:
            start = index_in_epoch
            if epochs_completed==0 and start==0:
                perm0 = np.arange(train_imgs.shape[2])
                np.random.shuffle(perm0)
                shuffle_imgs = train_imgs[..., perm0]
                shuffle_rsp = train_rsp[perm0]
            # go to next epoch
            if start + batch_size > train_imgs.shape[2]:
                # finish epoch
                epochs_completed += 1
                # get the rest examples in this epoch
                rest_num_examples = int(train_imgs.shape[2]) - start
                img_rest_part = shuffle_imgs[..., start:train_imgs.shape[2]]
                rsp_rest_part = shuffle_rsp[start:train_imgs.shape[2]]
                # shuffle the data
                perm = np.arange(train_imgs.shape[2])
                np.random.shuffle(perm)
                shuffle_imgs = train_imgs[..., perm]
                shuffle_rsp = train_rsp[perm]
                # start next epoch
                start = 0
                index_in_epoch = batch_size - rest_num_examples
                end = index_in_epoch
                img_new_part = shuffle_imgs[..., start:end]
                rsp_new_part = shuffle_rsp[start:end]
                img_batch = np.concatenate((img_rest_part,img_new_part), axis=2)
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
            _, summary, step_error, step_fpf = sess.run(
                                    [solver, merged, total_error, fpf],
                                    feed_dict={img: batch[0], rsp_: batch[1]})
            train_writer.add_summary(summary, iter_num)
            if (iter_num+1)%175==0:
                print 'Ep %s'%((iter_num+1)/175)
                print 'Training Error: %s'%(step_error)
                rsp_err = sess.run(error, feed_dict={img: batch[0],
                                                     rsp_: batch[1]})
                lap_err = sess.run(laplacian_error, feed_dict={img:batch[0],
                                                               rsp_: batch[1]})
                entropy_err = sess.run(entropy, feed_dict={img:batch[0],
                                                           rsp_: batch[1]})
                print 'Rsp error: %s'%(rsp_err)
                print 'Laplacian error: %s'%(lap_err)
                print 'Spatial entropy: %s'%(entropy_err)
                # model validation
                pred_val_rsp = np.zeros(175)
                for j in range(35):
                    part_rsp = sess.run(rsp,
                                feed_dict={img: val_imgs[(j*5):(j*5+5)],
                                           rsp_: val_rsp[(j*5):(j*5+5)]})
                    pred_val_rsp[(j*5):(j*5+5)] = part_rsp
                val_err = np.mean(np.square(pred_val_rsp - val_rsp))
                print 'Validation Error: %s'%(val_err)
                val_loss.append(val_err)
                if iter_num==174:
                    min_err = val_err
                else:
                    if (min_err - val_err) >= 0.0025:
                        min_err = val_err
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                # stop signal
                if patience_cnt > patience:
                    print 'Early stopping - step %s'%(iter_num)
                    # plot fpf
                    fig, ax = plt.subplots()
                    cax = ax.imshow(step_fpf, cmap='gray')
                    fig.colorbar(cax)
                    plt.savefig(os.path.join(vxl_dir,
                            'fpf_epoch%s.png'%((iter_num+1)/175)))
                    plt.close(fig)
                    # save model
                    saver.save(sess, os.path.join(vxl_dir, 'prf_model'),
                               global_step=(iter_num - (patience+1)*175))
                    saver.save(sess, os.path.join(vxl_dir, 'prf_model'),
                               global_step=iter_num, write_meta_graph=False)
                    # save final validation loss
                    with open(os.path.join(vxl_dir, 'val_loss.txt'), 'w+') as f:
                        val_idx = -1 * patience - 2
                        f.write('%s\n'%(val_loss[val_idx]))
                    break
            iter_num += 1

        train_writer.close()
        #test_writer.close()
    return

def tfprf_test_entropy(train_imgs, val_imgs, vxl_rsp, gabor_bank, vxl_dir):
    """Test laplacian regularized pRF model on test dataset."""
    # get image mask
    img_m = np.mean(train_imgs, axis=2)
    img_mask = imresize(img_m, (250, 250))
    # resized image value range: 0-255
    img_mask = np.reshape(img_mask<170, [-1])

    graph = tf.Graph()
    with graph.as_default():
        # vars for input data
        with tf.name_scope('input'):
            img = tf.placeholder("float", [None, 500, 500, 1], name='input-img')
            rsp_ = tf.placeholder("float", [None,], name='vxl-rsp')

        # var for feature pooling field
        with tf.name_scope('pooling-field'):
            fpf_kernel = tf.random_normal([1, 250, 250, 1], stddev=0.01)
            blur = np.array([[1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256],
                             [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                             [6.0/256, 24.0/256, 36.0/256, 24.0/256, 6.0/256],
                             [4.0/256, 16.0/256, 24.0/256, 16.0/256, 4.0/256],
                             [1.0/256,  4.0/256,  6.0/256,  4.0/256, 1.0/256]])
            blur = np.expand_dims(blur, 2)
            blur = np.expand_dims(blur, 3)
            fpf_kernel = tf.nn.conv2d(fpf_kernel, blur, strides=[1, 1, 1, 1],
                                      padding='SAME')
            fpf = tf.Variable(tf.reshape(fpf_kernel, [250, 250]), name='fpf')
            flat_fpf = tf.transpose(tf.boolean_mask(tf.reshape(fpf, (62500, 1)),
                                                    img_mask), [1, 0])

        # gabor features extraction
        with tf.name_scope('feature-extract'):
            feat_vtr = []
            for i in range(9):
                # config for the gabor filters
                gabor_real = np.expand_dims(gabor_bank['f%s_real'%(i+1)], 2)
                gabor_imag = np.expand_dims(gabor_bank['f%s_imag'%(i+1)], 2)
                rconv = tf.nn.conv2d(img, gabor_real, strides=[1, 2, 2, 1],
                                     padding='SAME')
                iconv = tf.nn.conv2d(img, gabor_imag, strides=[1, 2, 2, 1],
                                     padding='SAME')
                gabor_energy = tf.sqrt(tf.square(rconv) + tf.square(iconv))
                gabor_energy = tf.transpose(gabor_energy, perm=[1, 2, 3, 0])
                gabor_energy = tf.boolean_mask(tf.reshape(gabor_energy,
                                                        [62500, -1]), img_mask)
                # get feature summary from pooling field
                gabor_feat = tf.reshape(tf.matmul(flat_fpf, gabor_energy),
                                        (8, -1))
                feat_vtr.append(gabor_feat)
            # concatenate gabor features within fpf
            vxl_feats = tf.concat(feat_vtr, 0)

        # vars for feature weights
        with tf.name_scope('weighted-features'):
            b = tf.Variable(tf.constant(0.01, shape=[1]), name='bias')
            w = tf.Variable(tf.constant(0.01, shape=[1, 72]), name='weights')
            vxl_wt_feats = tf.matmul(w, vxl_feats)
            rsp = tf.reshape(vxl_wt_feats + b, [-1])

        # loss defination
        with tf.name_scope('loss'):
            # calculate fitting error
            error = tf.reduce_mean(tf.square(rsp - rsp_))
            # parameter regularization
            # laplacian regularization
            laplacian_kernel = np.array([[0, -1, 0],
                                         [-1, 4, -1],
                                         [0, -1, 0]])
            laplacian_kernel = np.expand_dims(laplacian_kernel, 2)
            laplacian_kernel = np.expand_dims(laplacian_kernel, 3)
            fpf_shadow = tf.expand_dims(tf.expand_dims(fpf, 0), 3)
            laplacian_error = tf.reduce_sum(tf.square(tf.nn.conv2d(fpf_shadow,
                                                         laplacian_kernel,
                                                         strides=[1, 1, 1, 1],
                                                         padding='VALID')))
            # get total error
            total_error = 10*error + 10*laplacian_error

    with tf.Session(graph=graph) as sess:
        # find the optimal model
        file_list = os.listdir(vxl_dir)
        file_list = [item for item in file_list if item[-5:]=='index']
        iter_num = [int(item.split('.')[0].split('-')[1]) for item in file_list]
        sel_iter_num = min(iter_num)
        model_path = os.path.join(vxl_dir, 'prf_model-%s'%(sel_iter_num))
        # load saved model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        # test on validation dataset
        input_imgs = val_imgs - np.expand_dims(img_m, 2)
        input_imgs = np.transpose(input_imgs, (2, 0, 1))
        input_imgs = np.expand_dims(input_imgs, 3)

        pred_val_rsp = np.zeros(120)
        for i in range(24):
            part_rsp = sess.run(rsp, feed_dict={img: input_imgs[(i*5):(i*5+5)],
                                                rsp_: vxl_rsp[(i*5):(i*5+5)]})
            pred_val_rsp[(i*5):(i*5+5)] = part_rsp
        val_err = np.mean(np.square(pred_val_rsp - vxl_rsp))
        print 'Validation Error: %s'%(val_err)
        # save final validation loss
        with open(os.path.join(vxl_dir, 'test_loss.txt'), 'w+') as f:
            f.write('%s\n'%(val_err))
    return

def get_gabor_features(input_imgs, gabor_bank):
    """Get Gabor features from images"""
    # vars for input data
    img = tf.placeholder("float", [None, 500, 500, 1])

    # gabor features extraction
    feat_vtr = []
    for i in range(9):
        # config for the gabor filters
        gabor_real = np.expand_dims(gabor_bank['f%s_real'%(i+1)], 2)
        gabor_imag = np.expand_dims(gabor_bank['f%s_imag'%(i+1)], 2)
        real_conv = tf.nn.conv2d(img, gabor_real, strides=[1, 2, 2, 1],
                                 padding='SAME')
        imag_conv = tf.nn.conv2d(img, gabor_imag, strides=[1, 2, 2, 1],
                                 padding='SAME')
        gabor_energy = tf.sqrt(tf.square(real_conv) + tf.square(imag_conv))
        feat_vtr.append(gabor_energy)
    # concatenate gabor features from various channels
    gabor_feat = tf.concat(feat_vtr, 3)

    # graph config
    config = tf.ConfigProto()
    with tf.Session(config=config) as sess:
        sess.run(tf.initialize_all_variables())
        gabor_file = 'train_gabor_feat.memdat'
        fp = np.memmap(gabor_file, dtype='float32', mode='w+',
                       shape=(input_imgs.shape[2], 250, 250, 72))
        for i in range(input_imgs.shape[2]/10):
            x = input_imgs[..., (i*10):(i*10+10)]
            x = np.transpose(x, (2, 0, 1))
            x = np.expand_dims(x, 3)
            gf = sess.run(gabor_feat, feed_dict={img: x})
            fp[(i*10):(i*10+10)] = gf


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

    #-- general config
    subj_id = 1
    roi = 'v1'
    # directory config
    subj_dir = os.path.join(res_dir, 'vim1_S%s'%(subj_id))
    prf_dir = os.path.join(subj_dir, 'prf')
    roi_dir = os.path.join(prf_dir, roi)
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir, 0755)

    #-- parameter preparation
    gabor_bank_file = os.path.join(db_dir, 'gabor_kernels_small.npz')
    gabor_bank = np.load(gabor_bank_file)

    #-- load vim1 stimuli of training dataset
    train_stimuli_file = os.path.join(db_dir, 'train_stimuli.npy')
    train_imgs = np.load(train_stimuli_file)

    #-- get gabor features from stimuli
    #get_gabor_features(input_imgs, gabor_bank)

    #-- pRF model test bench
    val_stimuli_file = os.path.join(db_dir, 'val_stimuli.npy')
    val_imgs = np.load(val_stimuli_file)
    vxl_idx, train_ts, val_ts = dataio.load_vim1_fmri(db_dir, subj_id, roi=roi)
    ts_m = np.mean(train_ts, axis=1, keepdims=True)
    ts_s = np.std(train_ts, axis=1, keepdims=True)
    train_ts = (train_ts - ts_m) / (ts_s + 1e-5)
    ts_m = np.mean(val_ts, axis=1, keepdims=True)
    ts_s = np.std(val_ts, axis=1, keepdims=True)
    val_ts = (val_ts - ts_m) / (ts_s + 1e-5)
    # to test the model. select following voxels
    sel_vxl_idx = [93, 257, 262, 385, 409, 485, 511, 517, 518, 603, 614,
                   807, 819, 820, 822, 826, 871, 929, 953, 1211]
    for i in sel_vxl_idx[:1]:
        print 'Voxel %s - %s'%(i, vxl_idx[i])
        vxl_dir = os.path.join(roi_dir, 'voxel_%s'%(vxl_idx[i]))
        os.makedirs(vxl_dir, 0755)
        # load voxel fmri data
        vxl_rsp = train_ts[i]
        tfprf_laplacian(train_imgs, vxl_rsp, gabor_bank, vxl_dir)
        #tfprf_laplacian_entropy(train_imgs, vxl_rsp, gabor_bank, vxl_dir)
        vxl_rsp = val_ts[i]
        tfprf_test(train_imgs, val_imgs, vxl_rsp, gabor_bank, vxl_dir)
        #tfprf_test_entropy(train_imgs, val_imgs, vxl_rsp, gabor_bank, vxl_dir)

    #-- laplacian regularized pRF
    #vxl_idx, train_ts, val_ts = dataio.load_vim1_fmri(db_dir, subj_id, roi=roi)
    #ts_m = np.mean(train_ts, axis=1, keepdims=True)
    #ts_s = np.std(train_ts, axis=1, keepdims=True)
    #train_ts = (train_ts - ts_m) / (ts_s + 1e-5)
    ##for i in range(vxl_idx.shape[0]):
    #for i in range(100):
    #    print 'Voxel %s - %s'%(i, vxl_idx[i])
    #    vxl_dir = os.path.join(roi_dir, 'voxel_%s'%(vxl_idx[i]))
    #    os.makedirs(vxl_dir, 0755)
    #    # load voxel fmri data
    #    vxl_rsp = train_ts[i]
    #    #print 'Image data shape ',
    #    #print input_imgs.shape
    #    #print 'Voxel time point number',
    #    #print vxl_rsp.shape
    #    tfprf_laplacian(train_imgs, vxl_rsp, gabor_bank, vxl_dir)

    #-- get validation r^2
    #vxl_idx, train_ts, val_ts = dataio.load_vim1_fmri(db_dir, subj_id, roi=roi)
    #ts_m = np.mean(train_ts, axis=1, keepdims=True)
    #ts_s = np.std(train_ts, axis=1, keepdims=True)
    #train_ts = (train_ts - ts_m) / (ts_s + 1e-5)
    #val_r2 = np.zeros(vxl_idx.shape[0])
    #for i in range(vxl_idx.shape[0]):
    #    print 'Voxel %s - %s'%(i, vxl_idx[i])
    #    vxl_dir = os.path.join(roi_dir, 'voxel_%s'%(vxl_idx[i]))
    #    val_mse = open(os.path.join(vxl_dir, 'val_loss.txt'), 'r').readlines()
    #    val_mse = float(val_mse[0].strip())
    #    # calculate r^2
    #    val_rsp = train_ts[i, 1575:]
    #    ss_tol = np.var(val_rsp)
    #    print 'Total variance: %s'%(ss_tol)
    #    print 'MSE: %s'%(val_mse)
    #    r2 = 1.0 - val_mse * 1.0 / ss_tol
    #    val_r2[i] = r2
    #outfile = os.path.join(roi_dir, 'dl_prf_val_r2.npy')
    #np.save(outfile, val_r2)

    #-- test prf model on test dataset
    ## load vim1 stimuli of validation dataset
    #val_stimuli_file = os.path.join(db_dir, 'val_stimuli.npy')
    #val_imgs = np.load(val_stimuli_file)
    ## load fmri data
    #vxl_idx, train_ts, val_ts = dataio.load_vim1_fmri(db_dir, subj_id, roi=roi)
    #ts_m = np.mean(val_ts, axis=1, keepdims=True)
    #ts_s = np.std(val_ts, axis=1, keepdims=True)
    #val_ts = (val_ts - ts_m) / (ts_s + 1e-5)
    #for i in range(vxl_idx.shape[0]):
    #    print 'Voxel %s - %s'%(i, vxl_idx[i])
    #    vxl_dir = os.path.join(roi_dir, 'voxel_%s'%(vxl_idx[i]))
    #    vxl_rsp = val_ts[i]
    #    tfprf_test(train_imgs, val_imgs, vxl_rsp, gabor_bank, vxl_dir)

    #-- get r^2 on test dataset
    #vxl_idx, train_ts, val_ts = dataio.load_vim1_fmri(db_dir, subj_id, roi=roi)
    #test_r2 = np.zeros(vxl_idx.shape[0])
    #for i in range(vxl_idx.shape[0]):
    #    print 'Voxel %s - %s'%(i, vxl_idx[i])
    #    vxl_dir = os.path.join(roi_dir, 'voxel_%s'%(vxl_idx[i]))
    #    test_mse = open(os.path.join(vxl_dir, 'test_loss.txt'), 'r').readlines()
    #    test_mse = float(test_mse[0].strip())
    #    # calculate r^2
    #    print 'MSE: %s'%(test_mse)
    #    r2 = 1.0 - test_mse*1.0
    #    test_r2[i] = r2
    #outfile = os.path.join(roi_dir, 'dl_prf_test_r2.npy')
    #np.save(outfile, test_r2)

    #-- parameter preparation
    #gabor_bank_file = os.path.join(feat_dir, 'gabor_kernels.npz')
    #gabor_bank = np.load(gabor_bank_file)
    #vxl_coding_paras_file =os.path.join(prf_dir,'tfrecon','vxl_coding_wts.npz')
    #vxl_coding_paras = np.load(vxl_coding_paras_file)

    #-- test encoding model
    #print 'Select voxel index',
    #print vxl_coding_paras['vxl_idx']
    #img_file = os.path.join(root_dir, 'example_imgs.npy')
    #imgs = np.load(img_file)
    #model_test(imgs, gabor_bank, vxl_coding_paras)

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

    

