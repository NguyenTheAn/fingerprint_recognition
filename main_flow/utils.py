import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import sys
from skimage.morphology import thin, skeletonize
from hog import hog
import math
import scipy
from scipy import signal
from scipy import ndimage

width, height = 242, 341

ridge_segment_blksze = 16
ridge_segment_thresh = 0.1
gradient_sigma = 1
block_sigma = 7
orient_smooth_sigma = 7
ridge_freq_blksze = 38
ridge_freq_windsze = 5
min_wave_length = 5
max_wave_length = 15
angleInc = 3.0
ridge_filter_thresh = -3

def normalise(img):
    normed = (img - np.mean(img)) / (np.std(img))
    return normed

def ridge_segment(img):
    img = normalise(img)
    rows, cols = img.shape
    new_rows = np.int(ridge_segment_blksze * np.ceil((np.float(rows)) / (np.float(ridge_segment_blksze))))
    new_cols = np.int(ridge_segment_blksze * np.ceil((np.float(cols)) / (np.float(ridge_segment_blksze))))
    # new rows and cols will divisible by ridge_segment_blksze 
    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))
    padded_img[0:rows][:, 0:cols] = img
    # residual wil be fill by 0
    for i in range(0, new_rows, ridge_segment_blksze):
        for j in range(0, new_cols, ridge_segment_blksze):
            block = padded_img[i:i + ridge_segment_blksze][:, j:j + ridge_segment_blksze] # get block of image
            stddevim[i:i + ridge_segment_blksze][:, j:j + ridge_segment_blksze] = np.std(block) * np.ones(block.shape)
            # std dev of block update to stddevim
    stddevim = stddevim[0:rows][:, 0:cols] # get image with stddev in old shape
    _mask = stddevim > ridge_segment_thresh # if stddev > thresh : is ridge 
    # normalize ridge region with 0 mean and unit std
    mean_val = np.mean(img[_mask])
    std_val = np.std(img[_mask])
    _normim = (img - mean_val) / (std_val)

    return _normim, _mask

def gauss(x, y):
    ssigma = 1.0
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma))


def kernel_from_function(size, f):
    kernel = [[] for i in range(0, size)]
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2))
    return kernel

def ridge_orient(norimg):
    rows,cols = norimg.shape
    #Calculate image gradients.
    sze = np.fix(6*gradient_sigma)
    if np.remainder(sze,2) == 0:
        sze = sze+1

    sobelx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    sobely = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    Gx = signal.convolve2d(norimg, sobelx, mode='same')
    Gy = signal.convolve2d(norimg, sobely, mode='same')

    Gxx = np.power(Gx,2)
    Gyy = np.power(Gy,2)
    Gxy = Gx*Gy

    #Now smooth the covariance data to perform a weighted summation of the data.
    sze = np.fix(6*block_sigma)

    gauss = cv2.getGaussianKernel(np.int(sze), block_sigma)
    f = gauss * gauss.T

    Gxx = ndimage.convolve(Gxx,f)
    Gyy = ndimage.convolve(Gyy,f)
    Gxy = 2*ndimage.convolve(Gxy,f)

    # Analytic solution of principal direction
    denom = np.sqrt(np.power(Gxy,2) + np.power((Gxx - Gyy),2)) + np.finfo(float).eps

    sin2theta = Gxy/denom                   # Sine and cosine of doubled angles
    cos2theta = (Gxx-Gyy)/denom


    if orient_smooth_sigma:
        sze = np.fix(6*orient_smooth_sigma)
        if np.remainder(sze,2) == 0:
            sze = sze+1
        gauss = cv2.getGaussianKernel(np.int(sze), orient_smooth_sigma)
        f = gauss * gauss.T
        cos2theta = ndimage.convolve(cos2theta,f)                   # Smoothed sine and cosine of
        sin2theta = ndimage.convolve(sin2theta,f)                   # doubled angles

    return np.pi/2 + np.arctan2(sin2theta,cos2theta)/2

def frequest(blkim, blkor):

    rows, cols = np.shape(blkim)

    # Find mean orientation within the block. This is done by averaging the
    # sines and cosines of the doubled angles before reconstructing the
    # angle again.  This avoids wraparound problems at the origin.

    cosorient = np.mean(np.cos(2 * blkor))
    sinorient = np.mean(np.sin(2 * blkor))
    orient = math.atan2(sinorient, cosorient) / 2

    # Rotate the image block so that the ridges are vertical

    rotim = scipy.ndimage.rotate(blkim, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3,
                                    mode='nearest')

    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.

    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

    # Sum down the columns to get a projection of the grey values down
    # the ridges.

    proj = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(proj, ridge_freq_windsze, structure=np.ones(ridge_freq_windsze))

    temp = np.abs(dilation - proj)

    peak_thresh = 2

    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)

    rows_maxind, cols_maxind = np.shape(maxind)

    # Determine the spatial frequency of the ridges by divinding the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds,
    # the frequency image is set to 0

    if (cols_maxind < 2):
        return(np.zeros(blkim.shape))
    else:
        NoOfPeaks = cols_maxind
        waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
        if waveLength >= min_wave_length and waveLength <= max_wave_length:
            return(1 / np.double(waveLength) * np.ones(blkim.shape))
        else:
            return(np.zeros(blkim.shape))

def ridge_freq(norimg, mask, orientimg):
    rows, cols = norimg.shape
    freq = np.zeros((rows, cols))

    for r in range(0, rows - ridge_freq_blksze, ridge_freq_blksze):
        for c in range(0, cols - ridge_freq_blksze, ridge_freq_blksze):
            blkim = norimg[r:r + ridge_freq_blksze][:, c:c + ridge_freq_blksze]
            blkor = orientimg[r:r + ridge_freq_blksze][:, c:c + ridge_freq_blksze]

            freq[r:r + ridge_freq_blksze][:, c:c + ridge_freq_blksze] = frequest(blkim, blkor)

    _freq = freq * mask
    freq_1d = np.reshape(_freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    non_zero_elems_in_freq = freq_1d[0][ind]

    _mean_freq = np.mean(non_zero_elems_in_freq)
    _median_freq = np.median(non_zero_elems_in_freq)  # does not work properly

    _freq = _mean_freq * mask
    return _freq

def gabor_filter(norimg, orientimg, freqimg, kx, ky):
    im = np.double(norimg)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))

    freq_1d = np.reshape(freqimg, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.

    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

    unfreq = np.unique(non_zero_elems_in_freq)

    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.

    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky

    sze = np.int(np.round(3 * np.max([sigmax, sigmay])))

    x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

    reffilter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
        2 * np.pi * unfreq[0] * x)        # this is the original gabor filter

    filt_rows, filt_cols = reffilter.shape

    angleRange = np.int(180 / angleInc)

    gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

    for o in range(0, angleRange):
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.

        rot_filt = scipy.ndimage.rotate(reffilter, -(o * angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt

    # Find indices of matrix points greater than maxsze from the image
    # boundary

    maxsze = int(sze)

    temp = freqimg > 0
    validr, validc = np.where(temp)

    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze

    final_temp = temp1 & temp2 & temp3 & temp4

    finalind = np.where(final_temp)

    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)

    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orientimg / np.pi * 180 / angleInc)

    # do the filtering
    for i in range(0, rows):
        for j in range(0, cols):
            if (orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if (orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    finalind_rows, finalind_cols = np.shape(finalind)
    sze = int(sze)
    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]

        img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

    _binim = newim < ridge_filter_thresh
    return _binim

def enhance(img):
    norimg, mask = ridge_segment(img)
    oriented = ridge_orient(norimg)
    freq = ridge_freq(norimg, mask, oriented)
    binimg = gabor_filter(norimg, oriented, freq, 0.65, 0.65)
    return binimg

def pad_image(image):
    if isinstance(image,str):
        image = cv2.imread(image)
        
    h, w = image.shape
    ratio = max(width, height) / max(h, w)
    image = cv2.resize(image, fx = ratio, fy = ratio, dsize = None)
    h, w = image.shape
    inp_image = np.zeros((height, width))
    if w > width:
        w = width
    inp_image[:h, :w] = image[:h, :w]
    
    return inp_image.astype(np.uint8)

def remove_background_noise(img, kernel_size):
    kernel1 = np.zeros((kernel_size, kernel_size))
    kernel1[:, 0] = 1
    kernel2 = np.zeros((kernel_size, kernel_size))
    kernel2[:, -1] = 1
    kernel3 = np.zeros((kernel_size, kernel_size))
    kernel3[0, :] = 1
    kernel4 = np.zeros((kernel_size, kernel_size))
    kernel4[-1, :] = 1
    origin = img.copy()
    pad = kernel_size // 2
    img = np.pad(img, (pad, pad), 'constant', constant_values=0)
    for y in range(pad, img.shape[0]-pad):
        for x in range(pad, img.shape[1]-pad):
            roi = img[y-pad:y+pad+1, x-pad:x+pad+1]
            sum1 = 1 if np.count_nonzero(roi*kernel1) > 1 else 0
            sum3 = 1 if np.count_nonzero(roi*kernel3) > 1 else 0
            sum2 = 1 if np.count_nonzero(roi*kernel2) > 1 else 0
            sum4 = 1 if np.count_nonzero(roi*kernel4) > 1 else 0
            if sum1+sum2+sum3+sum4 < 2:
                origin[y-pad, x-pad] = 0
    return origin

def preprocess(img):
    out = enhance(img.copy())
    out = np.array(out, dtype=np.uint8)
    out_rm_bgr = remove_background_noise(out, 9)
    y_indices, x_indices = np.where(out_rm_bgr == 1)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    roi = out[y_min:y_max, x_min:x_max]
    return roi*255