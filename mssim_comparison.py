from pytorch_msssim import ssim, ms_ssim
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_skimage
import tensorflow as tf
import matlab.engine
import cv2
import os
# to diseable tensorflow gpu warnings which are irrelevant here
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

img1_path = "images/1_1.jpg"
img2_path = "images/1_2.jpg"

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
    ssim_list = eng.msssim(img1_path,img2_path)
    ms = ssim_list[0][-1]
    ssims = ssim_list[0][0:-1]
    return ms, ssims

ssim_pytorch, ms_ssim_pytorch = calculate_pytorch_ssims(img1_path, img2_path)
ssim_ski = calculate_ski_ssim(img1_path, img2_path)
ssim_tf, ms_ssim_tf = calculate_tf_ssims(img1_path, img2_path)
ms_ssim_matlab, ssim_list_matlab = calculate_matlab_ssim(img1_path, img2_path)


print("#########SSIM################")
print("Pytorch ssim: %f" % (ssim_pytorch))
print("Tensorflow ssim: %f" % (ssim_tf))
print("Skimage ssim: %f" % (ssim_ski))
print("Matlab ssim_list: " + str(ssim_list_matlab))

print("\n#########MS-SSIM################")
print("Pytorch ms_ssim: %f" % (ms_ssim_pytorch))
print("Tensorflow ms_ssim: %f" % (ms_ssim_tf))
print("Matlab ms_ssim: %f" % (ms_ssim_matlab))

