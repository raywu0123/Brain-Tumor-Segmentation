import os
import argparse
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from models import MODELS
from data.data_providers import DataProviders
from utils import parse_exp_id


def prediction_argparse():
    parser = argparse.ArgumentParser(description='Brain Tumor Segmentation Prediction')
    parser.add_argument(
        '-i',
        '--exp_id',
        type=str,
        help='The exp_id to be used.',
    )
    return parser


def flow(
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
    pred = np.reshape(pred, (-1, model.depth, model.height, model.width))
    # (image_id D, H, W) to (image_id, H, W, D)
    pred = np.transpose(pred, (0, 2, 3, 1))
    print("######prediction ends######")

    test_ids = data_provider.test_ids

    prediction_path = os.path.join(model_path, "prediction")
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    for idx, test_id in enumerate(test_ids):
        image = pred[idx]
        path = os.path.join(prediction_path, test_id)
        image = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(image, path[:-4] + '.nii.gz')
        print(path[:-4] + '.nii.gz')


def predict(args):
    model_id, data_id, time_stamp = parse_exp_id(args.exp_id)
    data_provider = DataProviders[data_id]
    get_model, fit_hyper_parameters = MODELS[model_id]
    model_path = os.environ.get('RESULT_DIR') + args.exp_id
    os.environ['EXP_ID'] = args.exp_id
    model = get_model(
        **data_provider.get_data_format(),
    )

    flow(
        data_provider=data_provider,
        model=model,
        fit_hyper_parameters=fit_hyper_parameters,
        model_path=model_path
    )


def main():
    load_dotenv('./.env')
    parser = prediction_argparse()
    args = parser.parse_args()

    predict(args)


if __name__ == '__main__':
    main()
