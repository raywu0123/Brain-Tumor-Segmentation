import os

from parser import brain_tumor_argparse
import numpy as np
from dotenv import load_dotenv
from models import MODELS
from data.data_providers import DataProviders

"""if args.do_comet:
    experiment = Experiment(
        api_key=COMET_ML_KEY,
        log_code=False,
        project_name=args.comet_project,
        workspace=args.comet_workspace,
        parse_args=False,
    )
"""


def flow(
        data_provider,
        model,
        model_path,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    """if args.do_comet:
        fit_hyper_parameters['experiment'] = experiment"""

    model.load(model_path)

    test_volumes = data_provider.get_all_data()
    del test_volumes['label']
    pred = model.predict(test_volumes, fit_hyper_parameters)
    ids = data_provider.train_ids + data_provider.test_ids

    prediction_path = os.path.join(model_path, "prediction")
    os.mkdir(prediction_path)

    #np.save(pred, model_path)
    for idx, id in enumerate(ids):
        batch_size = fit_hyper_parameters[batch_size]
        pred = np.reshape(pred, (len(idx), -1, -1, -1, -1))
        pred = np.transpose(pred, 1, 2, 0)
        output_image = pred[idx]
        np.save(output_image, prediction_path + id + 'npy')


def predict(args):
    data_provider = DataProviders[args.data_provider_id]
    get_model, fit_hyper_parameters = MODELS[args.model_id]
    model_path = args.model_path

    model = get_model(
        **data_provider.get_data_format(),
    )

    """if args.do_comet:
        experiment.log_multiple_params(vars(args))"""

    flow(
        data_provider=data_provider,
        model=model,
        fit_hyper_parameters=fit_hyper_parameters,
        model_path=model_path
    )


def main():
    load_dotenv('./.env')
    """RESULT_DIR = os.environ.get('RESULT_DIR')
    COMET_ML_KEY = os.environ.get('COMET_ML_KEY')"""

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
