# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
import sys
import caffe


def mat2feat(stimulus, layer, phrase):
    """Get features of `layer` derived from CNN."""
    caffe.set_mode_cpu()
    #caffe.set_mode_gpu()
    model_dir = r'/nfs/diskstation/workshop/huanglijie/caffe_models'

    # reorder the data shape: to NxHxWxC
    stimulus = np.transpose(stimulus, (0, 3, 2, 1))
    print 'stimulus size :', stimulus.shape

    stim_len = stimulus.shape[0]
    if phrase=='train':
        part = 10
    else:
        part = 1
    unit = stim_len / part
    for i in range(part):
        # resize to 227 x 227
        input_ = np.zeros((unit, 227, 227, 3), dtype=np.float32)
        print 'input size :', input_.shape
        print 'Resize input image ...'
        for ix, im in enumerate(stimulus[(i*unit):(i+1)*unit]):
            input_[ix] = caffe.io.resize_image(im.astype(np.float32),(227, 227))
        # reorder the data shape: to NxCxHxW
        input_ = np.transpose(input_, (0, 3, 1, 2))
        # RGB to BGR
        input_ = input_[:, ::-1]
        # substract mean
        mean_file = os.path.join(model_dir, 'python', 'caffe', 'imagenet',
                                 'ilsvrc_2012_mean.npy')
        mean_im = np.load(mean_file)
        # take center crop
        center = np.array((256, 256)) / 2.0
        crop = np.tile(center, (1, 2))[0] + np.concatenate(
                [-np.array([227, 227]) / 2.0, np.array([227, 227]) / 2.0])
        crop = crop.astype(int)
        mean_im = mean_im[:, crop[0]:crop[2], crop[1]:crop[3]]
        mean_im = np.expand_dims(mean_im, 0)
        input_ -= mean_im

        # feedforward
        caffenet_dir =os.path.join(model_dir,'models','bvlc_reference_caffenet')
        caffenet = caffe.Net(os.path.join(caffenet_dir, 'deploy.prototxt'),
                os.path.join(caffenet_dir,'bvlc_reference_caffenet.caffemodel'),
                caffe.TEST)
        feat_s = caffenet.blobs[layer].data.shape
        if len(feat_s)>2:
            feat = np.zeros((input_.shape[0], feat_s[1]*feat_s[2]*feat_s[3]),
                            dtype=np.float32)
        else:
            feat = np.zeros((input_.shape[0], feat_s[1]), dtype=np.float32)

        batch_unit = input_.shape[0] / 10
        for j in range(batch_unit):
            batch_input = input_[(j*10):(j+1)*10]
            caffenet.forward(data=batch_input)
            tmp = np.copy(caffenet.blobs[layer].data)
            feat[(j*10):(j+1)*10] = tmp.reshape(10, -1)
        del caffenet
        if phrase=='val':
            np.save('%s_sti_%s.npy'%(layer, phrase), feat)
        else:
            np.save('%s_sti_%s_%s.npy'%(layer, phrase, i), feat)


if __name__ == '__main__':
    """Main function."""
    # config parser
    stim_dir = r'/nfs/diskstation/workshop/huanglijie/brainCoding/stimulus/vim2'

    #-- convert mat to cnn features
    layers = ['fc6', 'fc7', 'fc8']
    data_type = ['train', 'val']
    for l in layers:
        for dt in data_type:
            print l, dt
            # load original stimulus data
            data_dic = {'train': '/st', 'val': '/sv'}
            tf = tables.open_file(os.path.join(stim_dir, 'Stimuli.mat'))
            stimulus = tf.get_node(data_dic[dt])[:]
            tf.close()
            mat2feat(stimulus, l, dt)

