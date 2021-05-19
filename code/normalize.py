def normalise(img):
    # chuan hoa anh ve trung binh 0 va do lech chuan 1
    normed = (img - np.mean(img)) / (np.std(img))
    return normed