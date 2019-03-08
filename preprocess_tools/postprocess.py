import os
from sys import argv

from image_utils import ImageProcessor, save_array_to_nii


if __name__ == '__main__':
    label_dir = argv[1]
    print(f'label_dir:{label_dir}')
    file_names = os.listdir(label_dir)

    result_dir = argv[2]
    print(f'result_dir:{result_dir}')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        os.mkdir(os.path.join(result_dir, 'label'))

    image_processor_path = argv[3]
    print(f'image_processor_path:{image_processor_path}')

    image_processor = ImageProcessor()
    image_processor.load(image_processor_path)

    for file_name in file_names:
        print(file_name)
        file_id = file_name.strip('.nii.gz')
        label_path = os.path.join(label_dir, file_name)

        postprocessed_label, label_obj = image_processor.postprocess(
            label_path=label_path,
            file_id=file_id,
        )
        save_array_to_nii(
            postprocessed_label,
            os.path.join(result_dir, 'label', f'{file_id}.nii.gz'),
            label_obj.affine,
        )
