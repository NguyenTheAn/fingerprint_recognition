def normalize_by_ridge(img):
    mean_val = np.mean(img[_mask])
    std_val = np.std(img[_mask])
    _normim = (img - mean_val) / (std_val)
    return _normim