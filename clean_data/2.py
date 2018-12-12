import pickle
import os
'''
this file get the report which nii file exists in the folder
need "filtered_report.pickle"
generate 'mri_name.pickle'
'''

file_path = "../../../register3_result/"
mri_name = []

with open('filtered_report.pickle', 'rb') as file:
    report = pickle.load(file)


for i in range(len(report)):
    # 3-id, 5-check, 6-report, 4-class
    date = report[i][5]
    date = date.split('-')
    target = date[0] + '.' + date[1] + '.' + date[2]

    nii_name = os.listdir(file_path + report[i][3])

    for name in nii_name:
        if "txt" in name:
            a = 0
        elif target in name:
            new_name = "./register3_result/" + report[i][3] + '/' + name
            temp = [report[i][3], target, name]
            mri_name.append(temp)
            print(temp)

file = open('mri_name.pickle', 'wb')
pickle.dump(mri_name, file)
file.close()
