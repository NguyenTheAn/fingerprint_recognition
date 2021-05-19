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