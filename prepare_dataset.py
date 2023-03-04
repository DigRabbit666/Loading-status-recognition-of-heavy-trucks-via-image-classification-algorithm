import os
from shutil import copyfile
from tqdm import tqdm
import numpy.random as random


random.seed(42)
src_train_path = r'E:\zhonghuan_door\classification_data'
auto_train_path = r'E:\zhonghuan_door\train'
auto_val_path = r'E:\zhonghuan_door\val'
# if not os.path.isdir(auto_train_path):
#     os.makedirs(auto_train_path)
# if not os.path.isdir(auto_val_path):
#     os.makedirs(auto_val_path)

print('Handling training data...')
for roor, dirs, files in os.walk(src_train_path, topdown=True):
    for sdir in dirs:
        files = os.listdir(os.path.join(src_train_path, sdir))
        chosen_files = int(len(files) * 0.8)
        chosen_indexes = random.choice(files, chosen_files, replace=False)
        for file in tqdm(files):
            file_path = os.path.join(src_train_path, sdir, file)
            if file in chosen_indexes:
                if not os.path.isdir(os.path.join(auto_train_path, sdir)):
                    os.makedirs(os.path.join(auto_train_path, sdir))
                dst_path = os.path.join(auto_train_path, sdir, file)
                copyfile(file_path, dst_path)
            else:
                if not os.path.isdir(os.path.join(auto_val_path, sdir)):
                    os.makedirs(os.path.join(auto_val_path, sdir))
                dst_path = os.path.join(auto_val_path, sdir, file)
                copyfile(file_path, dst_path)

# print('Handling validation data...')
# for roor, dirs, files in os.walk(src_val_path, topdown=True):
#     for sdir in dirs:
#         files = os.listdir(os.path.join(src_val_path, sdir))
#         for file in tqdm(files):
#             file_path = os.path.join(src_val_path, sdir, file)
#             dst_path = os.path.join(auto_val_path, sdir, file)
#             copyfile(file_path, dst_path)















