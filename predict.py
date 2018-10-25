import os
import nibabel as nib
from parser import brain_tumor_argparse
import numpy as np
from dotenv import load_dotenv
from models import MODELS
from data.data_providers import DataProviders


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
    del test_volumes['label']

    pred = model.predict(test_volumes, **fit_hyper_parameters)
    print("######prediction ends######")

    ids = data_provider.test_ids
    pred = np.reshape(pred, (len(ids), 200, 200, 200))
    pred = np.transpose(pred, (0, 2, 3, 1))
    prediction_path = os.path.join(model_path, "prediction")
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    #np.save(pred, model_path)
    for idx, id in enumerate(ids):
        image = pred[idx]
        path = os.path.join(prediction_path, id)
        image = nib.Nifti1Image(image, affine=np.eye(4))
        nib.save(image, path + '.nii.gz')
        print(id)

def predict(args):
    data_provider = DataProviders[args.data_provider_id]
    get_model, fit_hyper_parameters = MODELS[args.model_id]
    model_path = args.model_path
    print(data_provider)
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

    parser = brain_tumor_argparse()
    parser.add_argument(
        '--model_path',
        type=str,
        help='The trained model\'s file path.',
    )

    args = parser.parse_args()

    predict(args)


if __name__ == '__main__':
    main()
