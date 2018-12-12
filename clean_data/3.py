import os
import pickle
'''
this file get T1 files and T1+C files
need "mri_name.pickle"
generate 'TC.pickle', T1.pickle
'''

nii_file_path = "../../../register3_result/"
path = '../nii_file'

folder = os.path.exists(path)
if not folder:
    os.mkdir(path)

with open('mri_name.pickle', 'rb') as file:
    mri_name = pickle.load(file)

mri_name.sort()

T1_list = []
TC_list = []

for name in mri_name:
    if "T1" in name[2]:
        x = name[2].find(name[1])
        time = name[2][x + 11:x + 16]
        if "+" in name[2]:
            temp = [name[0], name[1], time, name[2]]
            TC_list.append(temp)
        else:
            temp = [name[0], name[1], time, name[2]]
            T1_list.append(temp)

file = open('TC.pickle', 'wb')
pickle.dump(TC_list, file)
file.close()

file = open('T1.pickle', 'wb')
pickle.dump(T1_list, file)
file.close()
