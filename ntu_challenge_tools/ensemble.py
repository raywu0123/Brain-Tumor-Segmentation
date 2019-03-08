# Ensemble script for NTU Competition

import os
from sys import argv

import nibabel as nib
import numpy as np
from preprocess_tools.image_utils import save_array_to_nii
from tqdm import tqdm


if __name__ == '__main__':
    result_path = argv[1]
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    prob_predict_folders = argv[2:]
    filenames = os.listdir(prob_predict_folders[0])

    for filename in tqdm(filenames):
        all_probs = [
            nib.load(os.path.join(folder, filename)).get_fdata()
            for folder in prob_predict_folders
        ]
        nii_obj = nib.load(os.path.join(prob_predict_folders[0], filename))
        mean_probs = np.mean(all_probs, axis=0)
        binary_pred = (mean_probs > 0.5).astype(float)
        save_path = os.path.join(result_path, filename)
        save_array_to_nii(binary_pred, save_path, nii_obj.affine)
