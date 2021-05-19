def ridge_freq(norimg, mask, orientimg):
    rows, cols = norimg.shape
    freq = np.zeros((rows, cols))
    # tinh tan so tren moi block
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