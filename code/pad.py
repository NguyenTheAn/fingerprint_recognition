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