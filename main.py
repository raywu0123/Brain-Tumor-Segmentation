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
from optimizers import OptimizerFactory

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
        auxiliary_data_providers,
        auxiliary_data_provider_ids,
        trainer: PytorchTrainer,
        fit_hyper_parameters=None,
        **kwargs,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    auxiliary_data_generators = [
        aux_data_provider.get_full_data_generator(**kwargs)
        for aux_data_provider in auxiliary_data_providers
    ]
    trainer.fit_generator(
        training_data_generator=data_provider.get_training_data_generator(**kwargs),
        validation_data_generator=data_provider.get_testing_data_generator(**kwargs),
        auxiliary_data_generators=auxiliary_data_generators,
        auxiliary_data_provider_ids=auxiliary_data_provider_ids,
        metric=data_provider.metric,
        **fit_hyper_parameters,
    )


def main():
    if args.do_comet:
        experiment.log_parameters(vars(args))

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

    auxiliary_data_providers = []
    auxiliary_data_formats = []
    for data_provider_id in args.auxiliary_data_provider_ids:
        get_data_provider, data_provider_parameters = data_provider_hub[data_provider_id]
        aux_data_provider = get_data_provider(data_provider_parameters)
        auxiliary_data_providers.append(aux_data_provider)
        auxiliary_data_formats.append(aux_data_provider.data_format)

    get_model, fit_hyper_parameters = ModelHub[args.model_id]
    model = get_model(
        data_format=data_provider.data_format,
        loss_function_id=args.loss_function_id,
        clip_grad=args.clip_grad,
        optim_batch_steps=args.optim_batch_steps,
        auxiliary_data_formats=auxiliary_data_formats,
    )

    optimizer_factory = OptimizerFactory()
    optimizer, scheduler = optimizer_factory(
        model_parameters=model.parameters(),
        dataset_size=len(data_provider),
        optimizer_type=args.optimizer_type,
        lr=args.lr,
        epoch_milestones=args.epoch_milestones,
        gamma=args.gamma,
    )
    trainer = PytorchTrainer(
        model=model,
        dataset_size=len(data_provider),
        comet_experiment=experiment,
        profile=args.profile,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    flow(
        data_provider=data_provider,
        auxiliary_data_providers=auxiliary_data_providers,
        trainer=trainer,
        fit_hyper_parameters=fit_hyper_parameters,
        **vars(args),
    )


if __name__ == '__main__':
    main()
