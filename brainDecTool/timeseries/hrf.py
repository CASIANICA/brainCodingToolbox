# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from scipy.stats import gamma

def biGammaHRF(times):
    """Return values for HRF at given times. Unit of time: second.
    This HRF is derived from the sum of two Gamma function, it starts at
    zero, and gets to zero sometime brfore 35 seconds.
    """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

def vistaBiGammaHRF(times):
    """Return values for HRF at given times. Unit of time: second.
    This HRF is derived from VISTA software.
    """
    d1 = 5.4
    alpha1 = 5.98
    beta1 = 0.9
    c = 0.35
    d2 = 10.8
    alpha2 = 11.97
    beta2 = 0.9
    g = lambda t: ((t/d1)**alpha1)*np.exp((d1-t)/beta1)-c*((t/d2)**alpha2)*np.exp((d2-t)/beta2)
    return np.array([g(i) for i in times])

