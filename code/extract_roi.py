def extract_roi(img):
    y_indices, x_indices = np.where(img == 1)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    roi = out[y_min:y_max, x_min:x_max]
    return roi