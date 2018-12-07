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

from models import ModelHub
from data.data_providers import DataProviderHub
from utils import parse_exp_id
from trainers.pytorch_trainer import PytorchTrainer

load_dotenv('./.env')
RESULT_DIR = os.environ.get('RESULT_DIR')
COMET_ML_KEY = os.environ.get('COMET_ML_KEY')
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)

experiment = Experiment(
    api_key=COMET_ML_KEY,
    log_code=False,
    project_name=args.comet_project,
    workspace=args.comet_workspace,
    parse_args=False,
) if args.do_comet else None


def flow(
        data_provider,
        trainer,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    trainer.fit_generator(
        training_data_generator=data_provider.get_training_data_generator(),
        validation_data_generator=data_provider.get_testing_data_generator(),
        metric=data_provider.metric,
        **fit_hyper_parameters,
    )


def main():
    if args.do_comet:
        experiment.log_multiple_params(vars(args))

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

    data_provider_hub = DataProviderHub()
    get_data_provider, data_provider_parameters = data_provider_hub[args.data_provider_id]
    data_provider = get_data_provider(data_provider_parameters)

    get_model, fit_hyper_parameters = ModelHub[args.model_id]
    model = get_model(data_provider.data_format)
    if args.checkpoint_dir is not None:
        model.load(args.checkpoint_dir)

    trainer = PytorchTrainer(
        model=model,
        comet_experiment=experiment
    )

    flow(
        data_provider=data_provider,
        trainer=trainer,
        fit_hyper_parameters=fit_hyper_parameters,
    )


if __name__ == '__main__':
    main()
