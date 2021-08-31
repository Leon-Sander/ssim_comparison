from pytorch_msssim import ssim, ms_ssim
from matplotlib import pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_skimage
from online_codes import sp_ssim
import tensorflow as tf
import matlab.engine
import cv2
import os
import logging
import sys
# to diseable tensorflow gpu warnings which are irrelevant here
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.basicConfig(format='%(message)s',filename='output.log', level=logging.INFO)
img1_path = "images/1_1.jpg"
img2_path = "images/2_1.jpg"

#img1_path = "images/black.png"
#img2_path = "images/white.png"
#img2_path = img1_path
logging.info("img1: " + img1_path + ", img2: " + img2_path)

def calculate_pytorch_ssims(img1_path, img2_path):
    #https://github.com/VainF/pytorch-msssim/blob/master/tests/tests_loss.py
    img1 = np.array(cv2.imread(img1_path)).transpose(2, 0, 1)
    img1 = np.expand_dims(img1, axis=0)
    img1 = torch.from_numpy(img1).float()

    img2 = np.array(cv2.imread(img2_path)).transpose(2, 0, 1)
    img2 = np.expand_dims(img2, axis=0)
    img2 = torch.from_numpy(img2).float()

    # X: (N,3,H,W) a batch of non-negative RGB images (0~255)
    # Y: (N,3,H,W)  
    # set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
    ssim_pytorch = ssim( img1, img2, data_range=255, size_average=True) # return a scalar
    ms_ssim_pytorch = ms_ssim( img1, img2, data_range=255, size_average=True )

    return ssim_pytorch, ms_ssim_pytorch

def calculate_ski_ssim(img1_path, img2_path):
    #https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    #shape has to be H,W,3
    img1 = np.array(cv2.imread(img1_path))
    img2 = np.array(cv2.imread(img2_path))


    ssim_ski = ssim_skimage(img1, img2, data_range=255, multichannel=True)
    return ssim_ski

def calculate_tf_ssims(img1_path, img2_path):
    #https://www.tensorflow.org/api_docs/python/tf/image/ssim
    #shape has to be H,W,3
    _MSSSIM_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    img1 = np.array(cv2.imread(img1_path))
    img2 = np.array(cv2.imread(img2_path))
    ssim_tf = tf.image.ssim(img1, img2, 255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    ms_ssim_tf = tf.image.ssim_multiscale(img1, img2, 255, power_factors=_MSSSIM_WEIGHTS, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    return ssim_tf, ms_ssim_tf

def calculate_matlab_ssim(img1_path,img2_path):
    
    eng = matlab.engine.start_matlab()
    ssim_list, img1_grayscale, img2_grayscale = eng.msssim(img1_path,img2_path,nargout=3)
    ms = ssim_list[0][-1]
    ssim = ssim_list[0][0]
    return ms, ssim, img1_grayscale, img2_grayscale

def calculate_signal_processing_msssim(img1_path,img2_path):
    #https://mubeta06.github.io/python/sp/_modules/sp/ssim.html
    ssim_map, msssim_metric = sp_ssim.calculate_metrics(img1_path, img2_path)
    return msssim_metric

def calculate_all_metrics(img1_path, img2_path):

    ssim_pytorch, ms_ssim_pytorch = calculate_pytorch_ssims(img1_path, img2_path)
    ssim_ski = calculate_ski_ssim(img1_path, img2_path)
    ssim_tf, ms_ssim_tf = calculate_tf_ssims(img1_path, img2_path)
    ms_ssim_matlab, ssim_matlab, img1_grayscale, img2_grayscale = calculate_matlab_ssim(img1_path, img2_path)
    ms_ssim_sp = calculate_signal_processing_msssim(img1_path,img2_path)
    return ssim_pytorch, ms_ssim_pytorch, ssim_ski, ssim_tf, ms_ssim_tf, ms_ssim_matlab, ssim_matlab, ms_ssim_sp

def calculate_all_metrics_and_grayscale(img1_path, img2_path):

    ssim_pytorch, ms_ssim_pytorch = calculate_pytorch_ssims(img1_path, img2_path)
    ssim_ski = calculate_ski_ssim(img1_path, img2_path)
    ssim_tf, ms_ssim_tf = calculate_tf_ssims(img1_path, img2_path)
    ms_ssim_matlab, ssim_matlab, img1_grayscale, img2_grayscale = calculate_matlab_ssim(img1_path, img2_path)
    ms_ssim_sp = calculate_signal_processing_msssim(img1_path,img2_path)
    return img1_grayscale, img2_grayscale ,ssim_pytorch, ms_ssim_pytorch, ssim_ski, ssim_tf, ms_ssim_tf, ms_ssim_matlab, ssim_matlab, ms_ssim_sp 

def print_all_metrics(ssim_pytorch, ms_ssim_pytorch, ssim_ski, ssim_tf, ms_ssim_tf, ms_ssim_matlab, ssim_matlab, ms_ssim_sp):
    print("#########SSIM################")
    print("Pytorch ssim: %f" % (ssim_pytorch))
    print("Tensorflow ssim: %f" % (ssim_tf))
    print("Skimage ssim: %f" % (ssim_ski))
    print("Matlab ssim: %f" % (ssim_matlab))

    print("\n#########MS-SSIM################")
    print("Pytorch ms_ssim: %f" % (ms_ssim_pytorch))
    print("Tensorflow ms_ssim: %f" % (ms_ssim_tf))
    print("Matlab ms_ssim: %f" % (ms_ssim_matlab))
    print("Signal Processing ms_ssim: %f" % (ms_ssim_sp))

def log_all_metrics(ssim_pytorch, ms_ssim_pytorch, ssim_ski, ssim_tf, ms_ssim_tf, ms_ssim_matlab, ssim_matlab, ms_ssim_sp):
    logging.info("#########SSIM################")
    logging.info("Pytorch ssim: %f" % (ssim_pytorch))
    logging.info("Tensorflow ssim: %f" % (ssim_tf))
    logging.info("Skimage ssim: %f" % (ssim_ski))
    logging.info("Matlab ssim: %f" % (ssim_matlab))

    logging.info("\n#########MS-SSIM################")
    logging.info("Pytorch ms_ssim: %f" % (ms_ssim_pytorch))
    logging.info("Tensorflow ms_ssim: %f" % (ms_ssim_tf))
    logging.info("Matlab ms_ssim: %f" % (ms_ssim_matlab))
    logging.info("Signal Processing ms_ssim: %f" % (ms_ssim_sp))
    logging.info("\n")

def plt_results(img1_path,img2_path, img1_grayscale, img2_grayscale ,ssim_pytorch, ms_ssim_pytorch, ssim_ski, ssim_tf, ms_ssim_tf, ms_ssim_matlab, ssim_matlab, ms_ssim_sp):
    img1 = np.array(cv2.imread(img1_path))
    img2 = np.array(cv2.imread(img2_path))

    ssim_metrics = [float(ssim_pytorch), float(ssim_tf), ssim_matlab, ssim_ski]
    ssim_names = ['pytorch','tf','matlab','ski']

    ms_ssim_metrics = [float(ms_ssim_pytorch), float(ms_ssim_tf), ms_ssim_matlab, ms_ssim_sp]
    ms_ssim_names = ['pytorch','tf','matlab','sp']

    fig, (ax0, ax5, ax1, ax2, ax3, ax4) = plt.subplots(1,6, figsize=(25,5))


    bars_ssim = ax0.bar(ssim_names, ssim_metrics)
    for bar in bars_ssim:
        yval = bar.get_height()
        ax0.text(bar.get_x(), yval + 0.01, round(yval,4),color='black', fontweight='bold')
    ax0.set_ylim([0,1])
    ax0.set_title('ssim comparison')

    bars_msssim = ax5.bar(ms_ssim_names, ms_ssim_metrics)
    for bar in bars_msssim:
        yval = bar.get_height()
        ax5.text(bar.get_x(), yval + 0.01, round(yval,4),color='black', fontweight='bold' )
    ax5.set_ylim([0,1])
    ax5.set_title('ms_ssim comparison')

    ax1.imshow(img1)
    ax1.set_title('img1')
    ax1.axis('off')

    ax2.imshow(img2)
    ax2.set_title('img2')
    #ax2.axis('off')

    ax3.imshow(img1_grayscale)
    ax3.set_title('img1_grayscale')
    ax3.axis('off')

    ax4.imshow(img2_grayscale)
    ax4.set_title('img2_grayscale')
    #ax4.axis('off')
#ssim_pytorch, ms_ssim_pytorch, ssim_ski, ssim_tf, ms_ssim_tf, ms_ssim_matlab, ssim_matlab, ms_ssim_sp = calculate_all_metrics(img1_path,img2_path)
#print_all_metrics(ssim_pytorch, ms_ssim_pytorch, ssim_ski, ssim_tf, ms_ssim_tf, ms_ssim_matlab, ssim_matlab, ms_ssim_sp)
#log_all_metrics(ssim_pytorch, ms_ssim_pytorch, ssim_ski, ssim_tf, ms_ssim_tf, ms_ssim_matlab, ssim_matlab, ms_ssim_sp)

