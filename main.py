import os
import datetime

from comet_ml import Experiment
from parser import brain_tumor_argparse
parser = brain_tumor_argparse()
args = parser.parse_args()
import random
random.seed(args.global_random_seed)

import numpy as np
np.random.seed(args.global_random_seed)

from dotenv import load_dotenv

from models import MODELS
from data.data_providers import DataProviders
from utils import parse_exp_id

load_dotenv('./.env')
RESULT_DIR = os.environ.get('RESULT_DIR')
COMET_ML_KEY = os.environ.get('COMET_ML_KEY')
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

if args.do_comet:
    experiment = Experiment(
        api_key=COMET_ML_KEY,
        log_code=False,
        project_name=args.comet_project,
        workspace=args.comet_workspace,
        parse_args=False,
    )


def flow(
        data_provider,
        model,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    if args.do_comet:
        fit_hyper_parameters['experiment'] = experiment

    if args.use_generator:
        model.fit_generator(
            training_datagenerator=data_provider.training_datagenerator,
            validation_datagenerator=data_provider.testing_datagenerator,
            metric=data_provider.metric,
            **fit_hyper_parameters,
        )
    elif args.use_dataloader:
        model.fit_dataloader(
            get_training_dataloader=data_provider.get_training_dataloader,
            get_validation_dataloader=data_provider.get_testing_dataloader,
            metric=data_provider.metric,
            **fit_hyper_parameters,
        )
    else:
        model.fit(
            training_data=data_provider.get_training_data(),
            validation_data=data_provider.get_testing_data(),
            metric=data_provider.metric,
        )


def main():
    if args.checkpoint_dir is not None:
        folder_name = os.path.basename(os.path.normpath(args.checkpoint_dir))
        model_id, data_provider_id, time_stamp = parse_exp_id(folder_name)
        args.model_id, args.data_provider_id = model_id, data_provider_id

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.environ['EXP_ID'] = \
        f'{args.model_id}_on_{args.data_provider_id}_{time_stamp}'
    os.mkdir(os.path.join(RESULT_DIR, os.environ.get('EXP_ID')))
    args.exp_id = os.environ.get('EXP_ID')
    print('EXP_ID:', os.environ.get('EXP_ID'))

    get_data_provider, data_provider_parameters = DataProviders[args.data_provider_id]
    data_provider = get_data_provider(data_provider_parameters)
    get_model, fit_hyper_parameters = MODELS[args.model_id]

    model = get_model(
        **data_provider.get_data_format(),
    )

    if args.do_comet:
        experiment.log_multiple_params(vars(args))

    flow(
        data_provider=data_provider,
        model=model,
        fit_hyper_parameters=fit_hyper_parameters,
    )


if __name__ == '__main__':
    main()
