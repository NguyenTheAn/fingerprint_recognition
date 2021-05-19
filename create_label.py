import os
import glob
import random
import numpy as np

paths = glob.glob("validset/*/*.npy")

db_dict = dict()

for path in paths:
    name = path.split(("/"))[-1]
    sub = name.split("_")[0]
    db = path.split(("/"))[-2]
    if db not in db_dict.keys():
        db_dict[db] = {sub:[name]}
    elif sub not in db_dict[db].keys():
        db_dict[db][sub] = [name]
    else:
        db_dict[db][sub].append(name)
        
########## generate positive pair
file = open("valid_labels.txt", 'w')
for db in db_dict.keys():
    subjects = db_dict[db]
    for sub in subjects.keys():
        arr = subjects[sub]
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                file.write(f"{db}/{arr[i]} {db}/{arr[j]} 1\n")

db_list = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]

########## generate negative pair
for db in db_dict.keys():
    subjects = db_dict[db]
    for sub in subjects.keys():
        arr = subjects[sub]
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                id = random.randrange(0, 4)
                subjects_ = db_dict[db_list[id]]
                sub_ = random.choice(list(subjects_.keys()))
                while(sub_ == sub):
                    sub_ = random.choice(list(subjects_.keys()))
                arr_ = subjects_[sub_]
                file.write(f"{db}/{arr[i]} {db_list[id]}/{random.choice(arr_)} 0\n")
                # print(arr[i], random.choice(arr_))

file.close()