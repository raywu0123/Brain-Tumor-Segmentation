import os
from dotenv import load_dotenv

from parser import brain_tumor_argparse
parser = brain_tumor_argparse()
args = parser.parse_args()

from trainers.pytorch_trainer import PytorchTrainer
from models import ModelHub
from data.data_providers import DataProviderHub
from utils import parse_exp_id

load_dotenv('./.env')


def flow(
        data_provider,
        trainer,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    trainer.predict_on_generator(
        data_generator=data_provider.get_testing_data_generator(random=False),
        save_base_dir=os.path.join(args.checkpoint_dir, f'{args.data_provider_id}'),
        metric=data_provider.metric,
        **fit_hyper_parameters,
    )


if __name__ == '__main__':
    exp_id = os.path.basename(os.path.normpath(args.checkpoint_dir))
    model_id, data_id, time_stamp = parse_exp_id(exp_id)
    print(f'model_id: {model_id}',
          f'data_id: {data_id}',
          f'time_stamp: {time_stamp}',
          )
    os.environ['EXP_ID'] = exp_id

    data_provider_hub = DataProviderHub()
    get_data_provider, data_provider_parameters = data_provider_hub[args.data_provider_id]
    data_provider = get_data_provider(data_provider_parameters)

    get_model, fit_hyper_parameters = ModelHub[model_id]
    model = get_model(data_provider.data_format)

    trainer = PytorchTrainer(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
    )

    flow(
        data_provider=data_provider,
        trainer=trainer,
        fit_hyper_parameters=fit_hyper_parameters,
    )
