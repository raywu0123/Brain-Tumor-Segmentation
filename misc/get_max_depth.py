import os
import nibabel as nib
import numpy as np

STRUCTSEG_DIR = "/shares/Public/rays_data/StructSeg2019"
dirs = ["/HaN_OAR", "/Naso_GTV", "/Thoracic_OAR", "/Lung_GTV"]

print(STRUCTSEG_DIR)

for dir in dirs:
    folderName = STRUCTSEG_DIR + dir
    print(folderName)
    ids = os.listdir(folderName)

    depth = 0
    for id in ids:
        fileName = folderName + "/" + id + "/data.nii"
        image_obj = nib.load(fileName)
        image = image_obj.get_fdata()
        image = np.transpose(image, (2, 0, 1))
        d = image.shape[0]
        if d > depth:
            depth = d
    print(depth)
