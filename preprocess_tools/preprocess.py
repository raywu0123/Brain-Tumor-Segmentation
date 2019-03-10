import os
from sys import argv

from image_utils import ImageProcessor, save_array_to_nii


if __name__ == '__main__':
    data_dir = argv[1]
    print(f'data_dir:{data_dir}')
    file_names = os.listdir(os.path.join(data_dir, 'image'))
    file_names = [f for f in file_names if not f.startswith('.')]

    mask_dir = argv[2]
    print(f'mask_dir:{mask_dir}')

    result_dir = argv[3]
    print(f'result_dir:{result_dir}')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        os.mkdir(os.path.join(result_dir, 'image'))
        os.mkdir(os.path.join(result_dir, 'label'))

    image_processor = ImageProcessor()
    for file_name in file_names:
        print(file_name)
        file_id = file_name.strip('.nii.gz')
        image_path = os.path.join(data_dir, 'image', file_name)
        label_path = os.path.join(data_dir, 'label', file_name)
        mask_path = os.path.join(mask_dir, file_name)

        (preprocessed_image, image_obj), (preprocessed_label, label_obj) = \
            image_processor.preprocess(
                image_path=image_path,
                label_path=label_path,
                mask_path=mask_path,
                file_id=file_id,
        )
        # np.save(
        #     os.path.join(result_dir, 'image', f'{file_id}.npy'),
        #     preprocessed_image,
        # )
        # np.save(
        #     os.path.join(result_dir, 'label', f'{file_id}.npy'),
        #     preprocessed_label,
        # )
        save_array_to_nii(
            preprocessed_image,
            os.path.join(result_dir, 'image', f'{file_id}.nii.gz'),
            image_obj.affine,
        )
        save_array_to_nii(
            preprocessed_label,
            os.path.join(result_dir, 'label', f'{file_id}.nii.gz'),
            label_obj.affine,
        )

    image_processor.save(result_dir)
