import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

def distance(fp_encodings, fp_to_compare):
    return np.linalg.norm(fp_encodings - fp_to_compare, axis=0)

validset = glob.glob("validset/*/*.npy")
validset.sort()
print(len(validset))

data = glob.glob("features/*/*.npy")
print(len(data))

wrong = 0
correct = 0

for valid_path in tqdm(validset):
    anchor = np.load(valid_path)
    distances = []
    for path in data:
        feature = np.load(path)
        distances.append(distance(anchor, feature))
    idx = np.argmin(np.array(distances))
    label = valid_path.split("/")[-1][:-6] + "_" + valid_path.split("/")[1]
    pred = data[idx].split("/")[-1][:-6] + "_" + data[idx].split("/")[1]
    if label != pred:
        wrong += 1
    else:
        correct += 1
print(correct/(correct+wrong))