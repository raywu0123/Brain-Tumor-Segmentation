import sys
import csv
import pickle
'''
this file dropout report for none MRI files and files with Spine and Chest

to run this file, it requir list.txt in the same folder,
and it will generate a file "filtered_report.pickle"

list.txt: file name of all mri
'''
csv.field_size_limit(sys.maxsize)
file_name = "list.txt"

# turn data_id to list for the first time #
data_id = []
with open(file_name) as file:
    for line in file:
        line = line.strip('\n')
        data_id.append(line)
# load if you have data_id.pickle
# with open('data_id.pickle', 'rb') as file:
#     data_id =pickle.load(file)

filtered_report = []

# read report
with open("../analyze_report/report.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        file_name = row[4]
        if "MRI" in file_name:
            if "Spine" in file_name:
                a = 0
            elif "CHEST" in file_name:
                a = 0
            else:
                target = row[3]
                x = target.find("病歷號")
                if x != -1:
                    idx = target[x + 43: x + 50]
                    name = str(idx)
                    if name in data_id:
                        print("find")
                        filtered_report.append(row)

# save file #
file = open('filtered_report.pickle', 'wb')
pickle.dump(filtered_report, file)
file.close()

# read file #
# with open('idx.pickle', 'rb') as file:
#     element_set =pickle.load(file)
