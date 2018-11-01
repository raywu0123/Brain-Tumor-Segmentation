# Ensemble script for NTU Competition

import os
from sys import argv

import nibabel as nib
from tqdm import tqdm


if __name__ == '__main__':
    label_path = argv[1]
    predict_path = argv[2]
    filenames = os.listdir(label_path)
    filenames = [f for f in filenames if not f.startswith('.')]
    false_count = 0
    for filename in tqdm(filenames):
        label_obj = nib.load(os.path.join(label_path, filename))
        pred_obj = nib.load(os.path.join(predict_path, filename))
        if label_obj.shape != pred_obj.shape:
            false_count += 1

    print(f'false_count: {false_count}')
