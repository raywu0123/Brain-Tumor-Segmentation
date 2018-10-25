import os
import nibabel as nib
import numpy as np
from dotenv import load_dotenv

from parser import brain_tumor_argparse
from models import MODELS
from data.data_providers import DataProviders
from utils import parse_exp_id


def flow(
        data_id,
        data_provider,
        model,
        model_path,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    model.load(model_path)

    test_volumes = data_provider.get_testing_data()

    pred = model.predict(test_volumes, **fit_hyper_parameters)
    # (N, C, D, H, W) to (image_id, D, H, W)
    pred = np.reshape(pred, (-1, model.data_depth, model.data_height, model.data_width))
    # (image_id D, H, W) to (image_id, H, W, D)
    pred = np.transpose(pred, (0, 2, 3, 1))
    print("######prediction ends######")

    test_ids = data_provider.test_ids

    prediction_path = '/shares/Public/prediction'
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)
    prediction_path = os.path.join(prediction_path, data_id)
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    for image, test_id in zip(pred, test_ids):
        path = os.path.join(prediction_path, test_id)
        image = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(image, os.path.splitext(path)[0] + '.nii.gz')
        print(os.path.splitext(path)[0] + '.nii.gz')


def predict(args):
    exp_id = os.path.basename(os.path.normpath(args.checkpoint_dir))
    model_id, data_id, time_stamp = parse_exp_id(exp_id)
    data_provider = DataProviders[data_id]
    get_model, fit_hyper_parameters = MODELS[model_id]
    model_path = args.checkpoint_dir
    os.environ['EXP_ID'] = exp_id
    model = get_model(
        **data_provider.get_data_format(),
    )

    flow(
        data_id=data_id,
        data_provider=data_provider,
        model=model,
        fit_hyper_parameters=fit_hyper_parameters,
        model_path=model_path
    )


def main():
    load_dotenv('./.env')
    parser = brain_tumor_argparse()
    args = parser.parse_args()

    predict(args)


if __name__ == '__main__':
    main()
