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
    # tinh dao ham cua anh bang bo loc sobel
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
    # tinh huong cua duong van
    sin2theta = Gxy/denom              
    cos2theta = (Gxx-Gyy)/denom
    if orient_smooth_sigma:
        # lam tron duong van
        sze = np.fix(6*orient_smooth_sigma)
        if np.remainder(sze,2) == 0:
            sze = sze+1
        gauss = cv2.getGaussianKernel(np.int(sze), orient_smooth_sigma)
        f = gauss * gauss.T
        cos2theta = ndimage.convolve(cos2theta,f)                  
        sin2theta = ndimage.convolve(sin2theta,f)                

    return np.pi/2 + np.arctan2(sin2theta,cos2theta)/2