import os
import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm
import fingerprint_enhancer
import fingerprint_feature_extractor
from skimage.morphology import skeletonize
from skimage import exposure
from skimage import feature
import sys
from skimage.morphology import thin, skeletonize
from main_flow.hog import hog
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
    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))
    padded_img[0:rows][:, 0:cols] = img
    for i in range(0, new_rows, ridge_segment_blksze):
        for j in range(0, new_cols, ridge_segment_blksze):
            block = padded_img[i:i + ridge_segment_blksze][:, j:j + ridge_segment_blksze]
            stddevim[i:i + ridge_segment_blksze][:, j:j + ridge_segment_blksze] = np.std(block) * np.ones(block.shape)
    stddevim = stddevim[0:rows][:, 0:cols]

    _mask = stddevim > ridge_segment_thresh
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
    sze = np.fix(6*block_sigma)
    gauss = cv2.getGaussianKernel(np.int(sze), block_sigma)
    f = gauss * gauss.T

    Gxx = ndimage.convolve(Gxx,f)
    Gyy = ndimage.convolve(Gyy,f)
    Gxy = 2*ndimage.convolve(Gxy,f)
    denom = np.sqrt(np.power(Gxy,2) + np.power((Gxx - Gyy),2)) + np.finfo(float).eps
    sin2theta = Gxy/denom              
    cos2theta = (Gxx-Gyy)/denom
    if orient_smooth_sigma:
        sze = np.fix(6*orient_smooth_sigma)
        if np.remainder(sze,2) == 0:
            sze = sze+1
        gauss = cv2.getGaussianKernel(np.int(sze), orient_smooth_sigma)
        f = gauss * gauss.T
        cos2theta = ndimage.convolve(cos2theta,f)                  
        sin2theta = ndimage.convolve(sin2theta,f)                

    return np.pi/2 + np.arctan2(sin2theta,cos2theta)/2

# def ridge_orient(norimg):
#     #Calculate image gradients.
#     rows,cols = norimg.shape
#     sobelx = np.array([
#         [1, 0, -1],
#         [2, 0, -2],
#         [1, 0, -1]
#     ])
#     sobely = np.array([
#         [1, 2, 1],
#         [0, 0, 0],
#         [-1, -2, -1]
#     ])
#     Gx = cv2.filter2D(norimg, -1, sobelx)
#     Gy = cv2.filter2D(norimg, -1, sobely)

#     result_img = np.zeros_like(norimg)

#     pad = block_sigma // 2
#     pad_Gx = np.pad(Gx, (pad, pad), 'constant', constant_values=0)
#     pad_Gy = np.pad(Gy, (pad, pad), 'constant', constant_values=0)

#     for y in range(0, norimg.shape[0]):
#         for x in range(0, norimg.shape[1]):
#             i = x+pad
#             j = y+pad
#             Gx_ = pad_Gx[j-pad:j+pad+1, i-pad:i+pad+1]
#             Gy_ = pad_Gy[j-pad:j+pad+1, i-pad:i+pad+1]
#             Vx = (2*Gx_*Gy_).sum()
#             Vy = (Gx_*Gy_*Gx_*Gy_).sum()
#             theta = 0.5*np.arctan(Vy/(Vx + 1e-8))
#             result_img[y, x] = theta    


#     angles = np.array(result_img)
#     cos_angles = np.cos(angles.copy()*2)
#     sin_angles = np.sin(angles.copy()*2)

#     kernel = np.array(kernel_from_function(5, gauss))
#     cos_angles = cv2.filter2D(cos_angles,-1, kernel)
#     sin_angles = cv2.filter2D(sin_angles,-1, kernel)

#     smooth_angles = 0.5*np.arctan(sin_angles/(cos_angles + 1e-8))

#     return smooth_angles + np.pi/2

def frequest(blkim, blkor):
    rows, cols = np.shape(blkim)
    cosorient = np.mean(np.cos(2 * blkor))
    sinorient = np.mean(np.sin(2 * blkor))
    orient = math.atan2(sinorient, cosorient) / 2
    rotim = scipy.ndimage.rotate(blkim, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3,
                                    mode='nearest')
    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]
    proj = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(proj, ridge_freq_windsze, structure=np.ones(ridge_freq_windsze))
    temp = np.abs(dilation - proj)
    peak_thresh = 2
    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)
    rows_maxind, cols_maxind = np.shape(maxind)
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
    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100
    unfreq = np.unique(non_zero_elems_in_freq)
    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky
    sze = np.int(np.round(3 * np.max([sigmax, sigmay])))
    x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))
    reffilter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
        2 * np.pi * unfreq[0] * x)    
    filt_rows, filt_cols = reffilter.shape
    angleRange = np.int(180 / angleInc)
    gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))
    for o in range(0, angleRange):
        rot_filt = scipy.ndimage.rotate(reffilter, -(o * angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt
    maxsze = int(sze)
    temp = freqimg > 0
    validr, validc = np.where(temp)
    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze
    final_temp = temp1 & temp2 & temp3 & temp4
    finalind = np.where(final_temp)
    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orientimg / np.pi * 180 / angleInc)
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

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

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

def extract_roi(img):
    y_indices, x_indices = np.where(img == 1)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    roi = out[y_min:y_max, x_min:x_max]
    return roi

def preprocess(img):
    out = enhance(img.copy())
    out = np.array(out, dtype=np.uint8)
    out_rm_bgr = remove_background_noise(out, 9)
    y_indices, x_indices = np.where(out_rm_bgr == 1)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    roi = out[y_min:y_max, x_min:x_max]
    return roi

db = sys.argv[1]

save_dir = "features_minutiae"

data_paths = glob.glob(f"fingerprints/{db}/*.tif")

merge_size = []

for path in tqdm(data_paths):
    frame = cv2.imread(path, 0)
    # inp_image = enhance(frame.copy())
    inp_image = preprocess(frame.copy())
    # inp_image_ = fingerprint_enhancer.enhance_Fingerprint(frame.copy())
    inp_image = np.array(inp_image, dtype=np.uint8)
    inp_image *= 255

    # Terminations, Bifurcations = fingerprint_feature_extractor.extract_minutiae_features(img)
    # FeaturesTerminations, FeaturesBifurcations = [], []
    # for i in Terminations:
    #     FeaturesTerminations.append([i.locX, i.locY, i.Orientation])
    # for i in Bifurcations:
    #     FeaturesBifurcations.append([i.locX, i.locY, i.Orientation])
    # break
    
    # inp_image = skeletonize(inp_image > 0)
    # inp_image = np.array(inp_image, dtype=np.uint8)
    # inp_image *= 255

    # plot_comparison(frame, skel, "")
    # plt.show()
    # skel = np.array(thin(inp_image > 0)).astype(np.int8)*255
    
    # inp_image = preprocess(img)

    # os.makedirs(os.path.join("processed", db), exist_ok=True)
    # cv2.imwrite(os.path.join("processed", db, os.path.basename(path)[:-4]) + ".png", inp_image)

    # inp_image = pad_image(inp_image)
    # inp_image[inp_image >= 128] = 255
    # inp_image[inp_image < 128] = 0

    # (H, hogImage) = feature.hog(inp_image, orientations=9, pixels_per_cell=(8, 8),
    #     cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2",
    #     visualize=True)
    # H = hog(inp_image)

    # os.makedirs(os.path.join(save_dir, db), exist_ok=True)
    # np.save(os.path.join(save_dir, db, os.path.basename(path)[:-4]) + ".npy", feature)
    # np.save(os.path.join(save_dir, db, os.path.basename(path)[:-4]) + ".npy", H)

#     hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
#     hogImage = hogImage.astype("uint8")
    cv2.imshow("origin", frame)
    cv2.imshow("image", inp_image)
    # cv2.imshow("HOGimage", hogImage)
    k = cv2.waitKey(0)
    if k == ord('s'):
        cv2.imwrite("image/origin.png", frame)
        cv2.imwrite("image/extract_roi.png", inp_image)
    if k == ord("q"):
        break
cv2.destroyAllWindows()

