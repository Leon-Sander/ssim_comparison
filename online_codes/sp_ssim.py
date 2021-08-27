#!/usr/bin/env python
"""Module providing functionality to implement Structural Similarity Image 
Quality Assessment. Based on original paper by Z. Whang
"Image Quality Assessment: From Error Visibility to Structural Similarity" IEEE
Transactions on Image Processing Vol. 13. No. 4. April 2004.
"""

import sys
import numpy
from scipy import signal
from scipy import ndimage
import numpy as np
import cv2
from icecream import ic
#from random import gauss
#from sp import gauss

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    #ic(window.shape)
    window = np.expand_dims(window, axis=0)
    #ic(window.shape)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = numpy.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    #ic(weight.shape)
    downsample_filter = numpy.ones((2, 2))/4.0
    #ic(downsample_filter.shape)
    downsample_filter = np.expand_dims(downsample_filter, axis=0)
    im1 = img1.astype(numpy.float64)
    im2 = img2.astype(numpy.float64)
    mssim = numpy.array([])
    mcs = numpy.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map=True)
        mssim = numpy.append(mssim, ssim_map.mean())
        mcs = numpy.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        #ic(filtered_im1.shape)
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (numpy.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))

def calculate_metrics(img1_path,img2_path):
    img1 = np.array(cv2.imread(img1_path)).transpose(2, 0, 1)
    img2 = np.array(cv2.imread(img2_path)).transpose(2, 0, 1).astype(np.uint8)
    ssim_metric = ssim(img1,img2, cs_map=True)
    msssim_metric = msssim(img1, img2)

    return ssim_metric, msssim_metric

'''def main():
    """Compute the SSIM index on two input images specified on the cmd line."""
    import pylab
    argv = sys.argv
    if len(argv) != 3:
        print(sys.stderr, 'usage: python -m sp.ssim image1.tif image2.tif')
        sys.exit(2)

    try:
        from PIL import Image
        img1 = numpy.asarray(Image.open(argv[1]))
        img2 = numpy.asarray(Image.open(argv[2]))
    except Exception as e:
        e = 'Cannot load images' + str(e)
        print(sys.stderr, e)

    ssim_map = ssim(img1, img2)
    ms_ssim = msssim(img1, img2)

    pylab.figure()
    pylab.subplot(131)
    pylab.title('Image1')
    pylab.imshow(img1, interpolation='nearest', cmap=pylab.gray())
    pylab.subplot(132)
    pylab.title('Image2')
    pylab.imshow(img2, interpolation='nearest', cmap=pylab.gray())
    pylab.subplot(133)
    pylab.title('SSIM Map\n SSIM: %f\n MSSSIM: %f' % (ssim_map.mean(), ms_ssim))
    pylab.imshow(ssim_map, interpolation='nearest', cmap=pylab.gray())
    pylab.show()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())'''