import numpy as np
from scipy import ndimage as nd
import nibabel as nib
import pickle
import os


def save_array_to_nii(np_array, save_path, affine=None):
    empty_header = nib.Nifti1Header()
    if affine is None:
        affine = np.diag([1, 2, 3, 1])
    new_nii = nib.Nifti1Image(np_array, affine, empty_header)
    nib.save(new_nii, save_path)


def get_boundaries(vector):
    assert(len(vector.shape) == 1)
    vector = vector.astype(int)
    return np.argmax(vector), np.argmax(vector[::-1])


def get_dims(mask):
    assert(len(mask.shape) == 3)
    assert(len(np.unique(mask)) == 2)
    left, right = get_boundaries(np.any(mask, axis=(1, 2)))
    front, back = get_boundaries(np.any(mask, axis=(0, 2)))
    return (left, right), (front, back)


def crop_image_and_label_with_mask(image, label, mask):
    mask_or_label = np.logical_or(label != 0, mask != 0).astype(int)
    dims = get_dims(mask_or_label)
    (left, right), (front, back) = dims
    if right == 0:
        right = -image.shape[0]
    if back == 0:
        back = -image.shape[1]

    cropped_image = image[left:-right, front:-back, :]
    cropped_label = label[left:-right, front:-back, :]
    return cropped_image, cropped_label, dims


def zoom_image(image, zooms, binary=False):
    image = nd.zoom(image, zoom=zooms)
    if binary:
        image = image > 0.5
        image = image.astype(float)
    return image


def crop_or_pad_single_dim(image, dim, target_len):
    if target_len > image.shape[dim]:
        pad_width = np.zeros((image.ndim, 2), dtype=int)
        pad_width[dim][0] = int(np.ceil((target_len - image.shape[dim]) / 2))
        pad_width[dim][1] = int(np.floor((target_len - image.shape[dim]) / 2))
        return np.pad(image, pad_width, mode='constant', constant_values=0.)
    else:
        new_shape = np.array(image.shape)
        new_shape[dim] = target_len
        new_image = np.zeros(new_shape)
        left_index = int(np.ceil((image.shape[dim] - target_len) / 2))
        right_index = image.shape[dim] - int(np.floor((image.shape[dim] - target_len) / 2))
        take_indices = range(left_index, right_index)
        new_image = np.take(image, take_indices, axis=dim)
        return new_image


def crop_or_pad_to_shape(image, target_shape):
    if image.ndim != len(target_shape):
        raise ValueError(
            f'Incompatible shapes: image: {image.shape}, target_shape: {target_shape}'
        )
    target_shape = np.array(target_shape)
    for i_dim in range(image.ndim):
        image = crop_or_pad_single_dim(image, i_dim, target_shape[i_dim])

    assert(np.all(image.shape == target_shape))
    return image


def pad_label_with_dims(label, dims):
    assert(label.ndim >= 2)
    assert(len(dims) == 2)
    pad_width = np.zeros((label.ndim, 2), dtype=int)
    pad_width[0] = dims[0]
    pad_width[1] = dims[1]
    return np.pad(label, pad_width, mode='constant', constant_values=0.)


def load_nii(file_path):
    image_obj = nib.load(file_path)
    image = image_obj.get_fdata()
    return image_obj, image


class ImageProcessor():
    def __init__(
        self,
        target_shape: [int] = (200, 200, 200),
    ):
        self.target_shape = target_shape
        self.all_standard_shapes = {}
        self.all_cropped_shapes = {}
        self.all_dims = {}

    def preprocess(self, image_path, label_path, mask_path, file_id):
        image_obj, image = load_nii(image_path)
        if os.path.isfile(label_path):
            label_obj, label = load_nii(label_path)
        else:
            label_obj, label = image_obj, np.zeros_like(image)
        mask_obj, mask = load_nii(mask_path)
        header = image_obj.header
        zooms = header.get_zooms()

        image, label, dims = crop_image_and_label_with_mask(image, label, mask)
        self.all_cropped_shapes[file_id] = image.shape
        self.all_dims[file_id] = dims

        image = zoom_image(image, zooms, binary=False)
        label = zoom_image(label, zooms, binary=True)
        self.all_standard_shapes[file_id] = image.shape

        image = crop_or_pad_to_shape(image, self.target_shape)
        label = crop_or_pad_to_shape(label, self.target_shape)
        return (image, image_obj), (label, label_obj)

    def postprocess(self, label_path, file_id):
        label_obj, label = load_nii(label_path)

        standard_shape = self.all_standard_shapes[file_id]
        label = crop_or_pad_to_shape(label, standard_shape)

        cropped_shape = self.all_cropped_shapes[file_id]
        cropped_shape = np.array(cropped_shape)
        zooms = cropped_shape / np.array(label.shape)
        label = zoom_image(label, zooms, binary=True)

        dims = self.all_dims[file_id]
        label = pad_label_with_dims(label, dims)
        return label, label_obj

    def save(self, result_dir):
        file_path = os.path.join(result_dir, 'image_processor.pkl')
        with open(file_path, 'wb+') as file:
            pickle.dump(
                (
                    self.target_shape,
                    self.all_standard_shapes,
                    self.all_cropped_shapes,
                    self.all_dims
                ),
                file,
            )
        print(f'ImageProcessor saved to {file_path}')

    def load(self, file_path):
        with open(file_path, 'rb') as file:
            (
                self.target_shape,
                self.all_standard_shapes,
                self.all_cropped_shapes,
                self.all_dims
            ) = pickle.load(file)
        print(f'ImageProcessor loaded from {file_path}')
