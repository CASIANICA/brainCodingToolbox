# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

# This script shows how to reconstruct images from features with a trained
# model 

import sys

# uncomment and change the path to make python use specific caffe version
#sys.path.insert(0, '/home/huanglijie/decov/caffe_invert_alexnet/python')
#print sys.path

import caffe
import numpy as np
import os
import patchShow
import scipy.misc
import scipy.io

print(caffe.__file__)

# set up the inputs for the net: 
batch_size = 10
image_size = (3,227,227)
images = np.zeros((batch_size,) + image_size, dtype='float32')

# use crops of the cat image as an example 
in_image = scipy.misc.imread('Cat.jpg')
for ni in range(images.shape[0]):
  images[ni] = np.transpose(in_image[ni:ni+image_size[1], ni:ni+image_size[2]], (2,0,1))
# mirror some images to make it a bit more diverse and interesting
images[::2,:] = images[::2,:,:,::-1]
  
# RGB to BGR, because this is what the net wants as input
data = images[:,::-1] 

# subtract the ImageNet mean
matfile = scipy.io.loadmat('ilsvrc_2012_mean.mat')
image_mean = matfile['image_mean']
topleft = ((image_mean.shape[0] - image_size[1])/2, (image_mean.shape[1] - image_size[2])/2)
image_mean = image_mean[topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
del matfile
data -= np.expand_dims(np.transpose(image_mean, (2,0,1)), 0) # mean is already BGR

#initialize the caffe
# replace by caffe.set_mode_cpu() to run on a CPU
#caffe.set_mode_cpu() 
# net to extract the features
caffenet = caffe.Net('../caffenet/caffenet_deploy.prototxt', '../caffenet/caffenet.caffemodel')
#caffenet = caffe.Net('../caffenet/caffenet_deploy.prototxt', '../caffenet/caffenet.caffemodel', caffe.TEST)
caffenet.set_mode_cpu() 

# run caffenet and extract the features
caffenet.forward(data=data)
feat = np.copy(caffenet.blobs['fc6'].data)
del caffenet

# run the reconstruction net
net = caffe.Net('../fc6/invert_alexnet_fc6_deploy.prototxt', '../fc6/invert_alexnet_fc6.caffemodel')
#net = caffe.Net('../fc6/invert_alexnet_fc6_deploy.prototxt', '../fc6/invert_alexnet_fc6.caffemodel', caffe.TEST)
generated = net.forward(feat=feat)
recon = generated['deconv0'][:,::-1,topleft[0]:topleft[0]+image_size[1], topleft[1]:topleft[1]+image_size[2]]
del net

print images.shape, recon.shape

# save results to a file
collage = patchShow.patchShow(np.concatenate((images, recon), axis=3), in_range=(-120,120))
scipy.misc.imsave('reconstructions.png', collage)

  
  
  
  
  
