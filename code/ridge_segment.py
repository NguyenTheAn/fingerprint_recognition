def ridge_segment(img):
    # phan doan anh van 
    img = normalise(img)
    rows, cols = img.shape
    new_rows = np.int(ridge_segment_blksze * np.ceil((np.float(rows)) / (np.float(ridge_segment_blksze))))
    new_cols = np.int(ridge_segment_blksze * np.ceil((np.float(cols)) / (np.float(ridge_segment_blksze))))
    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))
    padded_img[0:rows][:, 0:cols] = img
    # tinh do lech chuan tren tung khoi
    for i in range(0, new_rows, ridge_segment_blksze):
        for j in range(0, new_cols, ridge_segment_blksze):
            block = padded_img[i:i + ridge_segment_blksze][:, j:j + ridge_segment_blksze]
            stddevim[i:i + ridge_segment_blksze][:, j:j + ridge_segment_blksze] = np.std(block) * np.ones(block.shape)
    stddevim = stddevim[0:rows][:, 0:cols]

    # neu do lech chuan lon hon nguong thi do la duong van
    _mask = stddevim > ridge_segment_thresh

    # chuan hoa anh theo duong van duoc phan doan
    mean_val = np.mean(img[_mask])
    std_val = np.std(img[_mask])
    _normim = (img - mean_val) / (std_val)

    return _normim, _mask
