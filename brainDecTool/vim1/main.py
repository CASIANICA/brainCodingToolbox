# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import tables
import bob.ip.gabor

def get_gabor_features(img):
    """Get Gabor features from input image."""
    img = img.astype(np.float64)
    gwt = bob.ip.gabor.Transform()
    trafo_img = gwt(img)
    out_feat = np.zeros((500, 500, 40))
    for i in range(trafo_img.shape[0]):
        real_p = np.real(trafo_img[i, ...])
        imag_p = np.imag(trafo_img[i, ...])
        out_feat[..., i] = np.sqrt(np.square(real_p)+np.square(imag_p))
    return out_feat


if __name__ == '__main__':
    """Main function."""
    db_dir = r'/home/huanglijie/workingdir/brainDecoding/vim1'
    #db_dir = r'/nfs/public_dataset/publicData/brain_encoding/crcns/vim-1'
    prefix = {'train': 'Stimuli_Trn_FullRes', 'val': 'Stimuli_Val_FullRes'}

    data_type = 'val'

    if data_type == 'train':
        for i in range(15):
            s_time = time.time()
            img_file = os.path.join(db_dir, prefix['train']+'_%02d.mat'%(i+1))
            print 'Load file %s ...'%(img_file)
            tf = tables.open_file(img_file)
            imgs = tf.get_node('/stimTrn')[:]
            tf.close()
            # output matrix: row x col x channel x image-number
            print 'image size %s'%(imgs.shape[2])
            out_features = np.zeros((500, 500, 40, imgs.shape[2]))
            for j in range(imgs.shape[2]):
                x = imgs[..., j].T
                out_features[..., j] = get_gabor_features(x)
            out_file = prefix['train']+'_%02d_gabor_features.npy'%(i+1)
            np.save(out_file, out_features)
            print 'Iter %s costs %s'%(i+1, time.time()-s_time)
    else:
        s_time = time.time()
        img_file = os.path.join(db_dir, prefix['val']+'.mat')
        print 'Load file %s ...'%(img_file)
        tf = tables.open_file(img_file)
        imgs = tf.get_node('/stimVal')[:]
        # output matrix: row x col x channel x image-number
        out_features = np.zeros((500, 500, 40, imgs.shape[2]))
        for j in range(imgs.shape[2]):
            x = imgs[..., j].T
            out_features[..., j] = get_gabor_features(x)
        out_file = prefix['val']+'_gabor_features.npy'
        np.save(out_file, out_features)
        print 'Val images costs %s'%(time.time()-s_time)


