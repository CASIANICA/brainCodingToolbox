# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Scripts for generating Gabor wavelets
@author: cddu
"""

import numpy as np
import pandas as pd

def make_gaussian(center, sig, n_pix):
    """Make a picture of a circular gaussian blob.
    center is the center of the blob in pixels. center of image is (0,0) 
    sig is one std. of the gaussian (pixels)
    n_pix is the size of the picture of the gaussian blob. i.e., output will 
    be an 2D array that is n_pix-by-n_pix

    """
    if n_pix % 2 == 0:
        pix_min = -n_pix/2
        pix_max = -pix_min
    else:
        pix_min = -(n_pix-1)/2
        pix_max = -pix_min+1
    
    [Xm, Ym] = np.meshgrid(range(pix_min,pix_max), range(pix_min,pix_max));  
    
    x0 = center[0]
    y0 = center[1]
    
    Z = (1. / 2*np.pi*sig**2)
    
    return Z *np.exp(-((Xm-x0)**2 + (Ym-y0)**2) / (2*sig**2))
  
def make_2D_sinewave(freq, theta, phase, n_pix):
    """freq is cycles/image
    theta is in radians
    phase is in radians (0 pi)
    center is (x,y) in pixel coordinates
    n_pix is size of the kernel in pixels
    
    """
    vec = np.array([np.sin(theta), np.cos(theta)]).reshape((2,1))
    
    if n_pix % 2 == 0:
        pix_min = -n_pix/2
        pix_max = -pix_min
    else:
        pix_min = -(n_pix-1)/2
        pix_max = -pix_min+1
    [Xm, Ym] = np.meshgrid(range(pix_min,pix_max), range(pix_min,pix_max));
    proj = np.array([Xm.ravel(), Ym.ravel()]).T.dot(vec)
    # compute proportion of Xm for given orientation
    Dt = np.sin(proj/n_pix*freq*2*np.pi + phase)
    Dt = Dt.reshape(Xm.shape)
    return Dt
 
def make_gabor(freq, theta, phase, center, sig, n_pix):
  return make_2D_sinewave(freq,theta,phase,n_pix) * make_gaussian(center,sig,n_pix)

def make_complex_gabor(freq,theta, center, sig, n_pix):
    """
    make_complex_gabor(freq,theta, center, sig, n_pix)
    freq is spatial frequency in cycles/image
    theta is orientation in radians
    center is (x,y) in pixel coordinates. center of image is (0,0)
    sig is one std of the gaussian envelope (pixels)
    n_pix is size of the kernel in pixels
    
    """
    phase = 0
    on_gabor = make_gabor(freq, theta, phase, center, sig, n_pix)
    phase = np.pi/2.
    off_gabor = make_gabor(freq, theta, phase, center, sig, n_pix)
    return off_gabor + 1j*on_gabor

def make_gabor_table(orientations, deg_per_stimulus, cycles_per_deg,
                     freq_spacing='log', pix_per_cycle=2, cycles_per_radius=1,
                     diams_per_filter=2, complex_cell=True):
    """Generates a table of parameters used to apply a gabor transform.
    This approach assumes that the gabors will be of fixed size (in pixels),
    but the image, the gabors are applied to, are downsampled to effectively
    determining the spatial frequency of the gabor.

    Returns everything needed to construct a stack of gabor filters.

    gbr_table, pix_per_filter, cyc_per_filter, envelope_radius =
                        make_gabor_table(orientations, deg_per_stimulus,
                                         cycles_per_deg, freq_spacing='log',
                                         pix_per_cycle=2, cycles_per_radius=1,
                                         diams_per_filter=2)

    orientations     ~ number of linearly spaced orientations in [0, pi)
    deg_per_stimulus ~ given the stimulus size and viewing distance
    cycles_per_deg   ~ specify range of spatial frequencies as
                       (lowest, highest, number) is cyc/deg.
    freq_spacing     ~ log or linear. spacing of spatial frequencies.
    pix_per_cycle    ~ how many pixels will be used to depict one cycle.
                       default = 2, i.e., the Nyquist limit.
                       Nyquist = 2 is fine for vert. or horz. orientations,
                       but too jaggy (probably) for obliques, but if too high,
                       usually will require stimuli with larger than native
                       resolution.
    cycles_per_radius~ determines radius of gaussian envelop.
                       we specify how many cycles per radius (= one stdev of
                       gaussian envelope)
                       default = 1 = one cycle of the sinewave per std. of the
                       gaussian envelope.
    diams_per_filter ~ determines the size of the filter. default = 2 = 4std. of
                       the gaussian envelope.
    complex_cell     ~ default = True. if False, we include distinguish between
                       filters with 0 and pi/2 phase
    
    returns
    gbr_table      ~ a pandas table with details of each gabor filter
    pix_per_filter ~ number of pixels per filter.  a constant.
    cyc_per_filter ~ number of cycles per filter.  a constant.
    envelope_radius~ number of pixels needed for one std. of the gaussian
                     envelope. a constant.
            
    Note: depending on the deg_per_stimulus of your experiment, you will be
    limited to a certain range of spatial frequencies. If too low, the filter
    will be larger than the downsampled image, which is kind of stupid. If too
    high, the image will have to be upsampled to obtain the required number of
    pixels per cycle, which is also stupid. The "full" range will have a lowest
    frequency where the image is downsampled to the size of the filter, and a 
    highest frequecy where the image is not downsampled at all. The larger the
    number of pixels per cycle, the smaller this range will be.

    """
    oris = np.linspace(0, np.pi, num=orientations,
                       endpoint=False).reshape(orientations)
 
    if freq_spacing == 'log':
        cycles_per_deg = np.logspace(np.log10(cycles_per_deg[0]),
                                     np.log10(cycles_per_deg[1]),
                                     num=cycles_per_deg[2])
    elif freq_spacing == 'linear':
        cycles_per_deg = np.linspace(cycles_per_deg[0],
                                     cycles_per_deg[1],
                                     num=cycles_per_deg[2])
    elif freq_spacing == 'log2':
        # f_n = f_0 * 2**(-(n-1)/2)
        cycles_per_deg = np.array([cycles_per_deg[1]/(2.0**(i/2.0))
                                   for i in range(cycles_per_deg[2])])
    
    # Inferred from your choices
    # radius of gaussian envelope of gabor filters in deg.
    envelope_radius = cycles_per_radius * (1./cycles_per_deg)
    
    # radius of gaussian envelope of gabor filters in pixels
    envelope_radius_pix = pix_per_cycle * cycles_per_radius

    # given the radius per filter, this is how many degrees the
    # picture of the filter should be
    deg_per_filter = 2 * envelope_radius * diams_per_filter

    # given pix/cyc, here's the # of pixels per stimulus
    pixels_per_stimulus = pix_per_cycle * cycles_per_deg * deg_per_stimulus

    # given deg/filter and min pix/cyc, this is how big the filter
    # should be (in pixels)
    pix_per_filter = deg_per_filter * pix_per_cycle * cycles_per_deg
    
    # cycles per filter
    cycles_per_filter = 2 * cycles_per_radius * diams_per_filter

    metrics = {'cycles per deg.': cycles_per_deg,
               'pix per stimulus' : pixels_per_stimulus,
               'radius of Gauss. envelope (deg)': envelope_radius,
               'filter size (deg.)': deg_per_filter,
               'pix_per_filter': np.round(pix_per_filter).astype('int'),
               'cycles_per_filter': cycles_per_filter}
    
    freq_table = pd.DataFrame(metrics)
    if not complex_cell:
        freq_table['phase'] = 0
        other_freq_table = freq_table.copy()
        other_freq_table['phase'] = np.pi/2.
        freq_table = pd.concat([freq_table, other_freq_table],
                               axis=0, ignore_index=True)
        
    freq_table['orientation'] = oris[0]
    tmp_freq_table = freq_table.copy()
    for o in oris[1:]:
        tmp_freq_table['orientation'] = o
        freq_table = pd.concat([freq_table, tmp_freq_table],
                               axis=0, ignore_index=True)
 
    return (freq_table, pix_per_filter[0],
            cycles_per_filter, envelope_radius_pix) 

def make_gabor_stack(gbr_table, pix_per_filter, cycles_per_filter,
                     envelope_radius_pix, complex_cell=True, color_channels=1):
    """Generate Gabor banks based on gabor table."""
    # initialize
    filter_stack = np.zeros((gbr_table.shape[0], color_channels,
                            int(pix_per_filter), int(pix_per_filter)))
    if complex_cell:
        filter_stack = filter_stack + 1j
    
    # args to gaborme
    center = (0,0)
    freq = cycles_per_filter
    radius = np.float32(envelope_radius_pix)
    n_pix = pix_per_filter.astype('int')
    for ii, tx in enumerate(gbr_table.index):
        ori = gbr_table.loc[tx,'orientation'] 
        for c in range(color_channels):
            if complex_cell:
                filter_stack[ii, c, :, :] = make_complex_gabor(freq,ori, center,
                                                               radius, n_pix)
            else:
                ph = gbr_table.loc[ii, 'phase']
                filter_stack[ii, c, :, :] = make_gabor(freq, ori, ph, center,
                                                       radius, n_pix)
    return filter_stack

class gabor_filters_creating(object):
    """Class for creating Gabor filters."""
    def __init__(self, orientations, deg_per_stimulus, cycles_per_deg,
                 freq_spacing='log', pix_per_cycle=2, cycles_per_radius=1,
                 diams_per_filter=2, complex_cell=True, color_channels=1):
        self.number_of_orientations = orientations
        self.deg_per_stimulus = deg_per_stimulus
        self.lowest_freq = cycles_per_deg[0]
        self.highest_freq = cycles_per_deg[1]
        self.num_sp_freq = cycles_per_deg[2]
        self.freq_spacing = freq_spacing
        self.pix_per_cycle = pix_per_cycle
        self.cycles_per_radius = cycles_per_radius
        self.diams_per_filter = diams_per_filter
        self.complex_cell = complex_cell
        self.color_channels = color_channels
        (self.gbr_table,
         self.pix_per_filter,
         self.cycles_per_filter,
         self.envelope_radius_pix) = make_gabor_table(orientations,
                                                      deg_per_stimulus,
                                                      cycles_per_deg,
                                                      freq_spacing,
                                                      pix_per_cycle,
                                                      cycles_per_radius,
                                                      diams_per_filter,
                                                      complex_cell=self.complex_cell)
        self.filter_stack = make_gabor_stack(self.gbr_table,
                                             self.pix_per_filter,
                                             self.cycles_per_filter,
                                             self.envelope_radius_pix,
                                             color_channels=color_channels,
                                             complex_cell=self.complex_cell)

    def sensitivity(self, feat_dict, parameter):
        """Given a feature dictionary produced by current instance, returns mean
        response per parameter. It can be used to get spatial freq. or orientation
        or phase sensitivity of population.
        Returns param_values, mean_response for plotting like
        plot(param_values, mean_response)
        """
        param_group = self.gbr_table.groupby(parameter)
        mean_resp = np.zeros(len(param_group))
        param_values = np.zeros(len(param_group))
        ii = 0
        for name, grp in param_group:
            idx = grp.index
            mean_resp[ii] = np.mean(map(lambda x: np.mean(feat_dict[x]), idx))
            param_values[ii] = name
            ii += 1
        return param_values, mean_resp

