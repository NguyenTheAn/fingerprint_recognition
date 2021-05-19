import glob
import numpy as np
import os
import random

root_folder = "features"
save_dir = "validset"

dbs = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]
instances = ["101", "102", "103", "104", "105", "106", "107", "108", "109", "110"]

for db in dbs:
    for instance in instances:
        os.makedirs(f"{save_dir}/{db}", exist_ok=True)
        os.system(f"mv {root_folder}/{db}/{instance}_8.npy {save_dir}/{db}")
        os.system(f"mv {root_folder}/{db}/{instance}_1.npy {save_dir}/{db}")

    