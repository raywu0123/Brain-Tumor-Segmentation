import os
import nibabel as nib
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from parser import brain_tumor_argparse
parser = brain_tumor_argparse()
args = parser.parse_args()

from models import MODELS
from data.data_providers import DataProviders
from utils import parse_exp_id

load_dotenv('./.env')


def flow(
        data_provider_id,
        data_provider,
        model,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    test_volumes = data_provider.get_all_data()
    pred = model.predict(
        test_volumes,
        **{
            **fit_hyper_parameters,
            'verbose': True,
        }
    )

    # (N, C, D, H, W) to (image_id, D, H, W)
    pred = np.reshape(pred, (-1, model.data_depth, model.data_height, model.data_width))
    pred = (pred > 0.5).astype(float)
    # (image_id, D, H, W) to (image_id, H, W, D)
    pred = np.transpose(pred, (0, 2, 3, 1))

    test_ids = data_provider.train_ids + data_provider.test_ids
    
    prediction_path = os.path.join(args.checkpoint_dir, f'predict_on_{data_provider_id}')
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    for image, test_id in tqdm(zip(pred, test_ids)):
        test_id = test_id.strip('.npy').strip('.nii.gz')
        path = os.path.join(prediction_path, f'{test_id}.nii.gz')
        image = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(image, path)


if __name__ == '__main__':
    exp_id = os.path.basename(os.path.normpath(args.checkpoint_dir))
    model_id, data_id, time_stamp = parse_exp_id(exp_id)
    os.environ['EXP_ID'] = exp_id

    get_data_provider, data_provider_parameters = DataProviders[args.data_provider_id]
    data_provider = get_data_provider(data_provider_parameters)

    get_model, fit_hyper_parameters = MODELS[model_id]
    model = get_model(
        **data_provider.get_data_format(),
    )
    model.load(args.checkpoint_dir)

    flow(
        data_provider_id=args.data_provider_id,
        data_provider=data_provider,
        model=model,
        fit_hyper_parameters=fit_hyper_parameters,
    )
