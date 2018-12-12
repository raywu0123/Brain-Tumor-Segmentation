import os
import pickle
from shutil import copyfile
from tqdm import tqdm

'''
this file copy file to the right place and rename them
need "T1.pickle" or "T1+C.pickle"
'''

nii_file_path = "../../../register3_result/"
path = '../nii_file'

folder = os.path.exists(path)
if not folder:
    os.mkdir(path)

with open('T1.pickle', 'rb') as file:
    mri_name = pickle.load(file)

for name in tqdm(mri_name):
    dir_path = path + '/' + name[0] + '_' + name[1]
    folder = os.path.exists(dir_path)
    if not folder:
        os.mkdir(dir_path)

    copy_path = nii_file_path + name[0] + '/' + name[3]
    dst_name = dir_path + '/T1.nii.gz'
    copyfile(copy_path, dst_name)

    file_list = os.listdir(nii_file_path + name[0])
    for f in file_list:
        if name[2] in f and "target" in f:
            copy_path = nii_file_path + name[0] + '/' + f
            dst_name = dir_path + '/target.nii.gz'
            copyfile(copy_path, dst_name)
