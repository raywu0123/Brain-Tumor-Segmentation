import os
import nibabel as nib
from sys import argv
import numpy as np
# from tqdm import tqdm
# from collections import Counter
import scipy.ndimage as nd
# from bistiming import SimpleTimer
import pickle


def get_boundaries(vector):
    assert(len(vector.shape) == 1)
    vector = vector.astype(int)
    return np.argmax(vector), len(vector) - np.argmax(vector[::-1])


def get_dims(image):
    assert(len(image.shape) == 3)
    assert(len(np.unique(image)) == 2)
    left, right = get_boundaries(np.any(image, axis=(1, 2)))
    front, back = get_boundaries(np.any(image, axis=(0, 2)))
    return (left, right), (front, back)


def load_nii(path):
    image_obj = nib.load(file_path)
    image = image_obj.get_fdata()
    return image_obj, image


def save_array_to_nii(np_array, save_path, original_nii):
    affine = original_nii.affine
    empty_header = nib.Nifti1Header()
    new_nii = nib.Nifti1Image(np_array, affine, empty_header)
    nib.save(new_nii, save_path)


if __name__ == '__main__':
    data_dir = argv[1]
    print(f'data_dir:{data_dir}')
    file_names = os.listdir(os.path.join(data_dir, 'image'))

    mask_dir = argv[2]
    print(f'mask_dir:{mask_dir}')

    result_dir = argv[3]
    print(f'result_dir:{result_dir}')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        os.mkdir(os.path.join(result_dir, 'image'))
        os.mkdir(os.path.join(result_dir, 'label'))

    all_shapes = {}
    all_dims = {}
    all_zooms = {}
    for file_name in file_names:
        print(file_name)
        file_path = os.path.join(data_dir, 'image', file_name)
        image_obj, image = load_nii(file_path)
        header = image_obj.header
        zooms = header.get_zooms()

        file_path = os.path.join(data_dir, 'label', file_name)
        label_obj, label = load_nii(file_path)

        file_path = os.path.join(mask_dir, file_name)
        mask_obj, mask = load_nii(file_path)

        (left, right), (front, back) = get_dims(mask)
        cropped_image = image[left:right, front:back, :]
        cropped_label = label[left:right, front:back, :]

        cropped_image = nd.zoom(cropped_image, zoom=zooms)
        cropped_label = nd.zoom(cropped_label, zoom=zooms)
        cropped_label = cropped_label > 0.5
        cropped_label = cropped_label.astype(float)

        all_shapes[file_name] = cropped_image.shape
        all_dims[file_name] = ((left, right), (front, back))
        all_zooms[file_name] = zooms

        file_id = file_name.strip('.nii.gz')
        # np.save(os.path.join(result_dir, 'image', f'{file_id}.npy'), cropped_image)
        # np.save(os.path.join(result_dir, 'label', f'{file_id}.npy'), cropped_label)
        save_array_to_nii(
            cropped_image,
            os.path.join(result_dir, 'image', f'{file_id}.nii.gz'),
            image_obj,
        )
        save_array_to_nii(
            cropped_label,
            os.path.join(result_dir, 'label', f'{file_id}.nii.gz'),
            label_obj,
        )

    with open(os.path.join(result_dir, 'info.pkl'), 'wb+') as file:
        pickle.dump((all_shapes, all_dims, all_zooms), file)
