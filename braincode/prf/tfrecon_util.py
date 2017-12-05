# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import bob.ip.gabor


def get_gabor_kernels():
    # gabor bank generation
    gwt = bob.ip.gabor.Transform(number_of_scales=9)
    gwt.generate_wavelets(500, 500)
    gabor_real = np.zeros((500, 500, 72))
    gabor_imag = np.zeros((500, 500, 72))
    for i in range(72):
        w = bob.ip.gabor.Wavelet(resolution=(500, 500),
                        frequency=gwt.wavelet_frequencies[i])
        sw = bob.sp.ifft(w.wavelet.astype(np.complex128)) 
        gabor_real[..., i] = np.roll(np.roll(np.real(sw), 250, 0), 250, 1)
        gabor_imag[..., i] = np.roll(np.roll(np.imag(sw), 250, 0), 250, 1)
    np.savez('gabor_kernels', gabor_real=gabor_real, gabor_imag = gabor_imag)


if __name__ == '__main__':
    get_gabor_kernels()

