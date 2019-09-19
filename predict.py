import os
from dotenv import load_dotenv

from parser import brain_tumor_argparse
parser = brain_tumor_argparse()
args = parser.parse_args()

from tqdm import tqdm

from trainers.pytorch_trainer import PytorchTrainer
from models import ModelHub
from models.utils import summarize_logs
from data.data_providers import DataProviderHub
from utils import parse_exp_id, highlight_print

load_dotenv('./.env')


def flow(
        data_provider,
        trainer,
        args,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    get_data_generator_fns = {
        'full': data_provider.get_full_data_generator,
        'test': data_provider.get_testing_data_generator,
    }
    get_data_generator_fn = get_data_generator_fns[args.predict_mode]
    all_metric_dict = trainer.predict_on_generator(
        data_generator=get_data_generator_fn(random=False),
        save_base_dir=os.path.join(args.checkpoint_dir, f'{args.data_provider_id}'),
        metric=data_provider.metric,
        save_volume=args.save_volume,
        **fit_hyper_parameters,
    )
    all_metric_list = [all_metric_dict[key] for key in all_metric_dict.keys()]
    highlight_print(f'full average metric')
    print(summarize_logs(all_metric_list))

    if 'diagnosis' in data_provider.data_format:
        highlight_print('metric by class')
        data_generator = get_data_generator_fn(random=False)
        metric_list_by_diagnosis = categorize_by_diagnosis(all_metric_dict, data_generator)
        for key in metric_list_by_diagnosis.keys():
            print(f'diagnosis: {key}, {summarize_logs(metric_list_by_diagnosis[key])}')


def categorize_by_diagnosis(all_metric_dict, data_generator) -> dict:
    print(f'categorizing on {len(data_generator)} volumes...')
    categorized_dict = {}
    for _ in tqdm(range(len(data_generator))):
        batch_data = data_generator(batch_size=1)
        data_id = batch_data['data_ids'][0]
        diagnosis = batch_data['diagnosis'][0]
        if diagnosis not in categorized_dict:
            categorized_dict[diagnosis] = []
        categorized_dict[diagnosis].append(all_metric_dict[data_id])
    return categorized_dict


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

    auxiliary_data_providers = []
    auxiliary_data_formats = []
    for data_provider_id in args.auxiliary_data_provider_ids:
        get_data_provider, data_provider_parameters = data_provider_hub[data_provider_id]
        aux_data_provider = get_data_provider(data_provider_parameters)
        auxiliary_data_providers.append(aux_data_provider)
        auxiliary_data_formats.append(aux_data_provider.data_format)

    get_model, fit_hyper_parameters = ModelHub[model_id]
    model = get_model(
        data_format=data_provider.data_format,
        auxiliary_data_formats=auxiliary_data_formats,
    )

    trainer = PytorchTrainer(
        model=model,
        checkpoint_dir=args.checkpoint_dir,
        dataset_size=len(data_provider),
    )

    flow(
        data_provider=data_provider,
        trainer=trainer,
        args=args,
        fit_hyper_parameters=fit_hyper_parameters,
    )
