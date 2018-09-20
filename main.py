import os
import datetime

from parser import brain_tumor_argparse
parser = brain_tumor_argparse()
args = parser.parse_args()
import random
random.seed(args.global_random_seed)

import numpy as np
np.random.seed(args.global_random_seed)

from dotenv import load_dotenv

from models import MODELS
# from utils import MetricClass
from data.data_providers import DataProviders

load_dotenv('./.env')
RESULT_DIR = os.environ.get('RESULT_DIR')
if not os.path.exists(RESULT_DIR):
    os.mkdir(RESULT_DIR)


def flow(
        data_provider,
        model,
        fit_hyper_parameters=None,
    ):
    if fit_hyper_parameters is None:
        fit_hyper_parameters = {}

    # (x_train, y_train), (x_test, y_test) = data_provider.get_training_data()
    # model.fit(
    #     training_data=(x_train, y_train),
    #     validation_data=(x_test, y_test),
    #     **fit_hyper_parameters,
    # )
    #
    # prediction = model.predict(x_test)
    # metric = MetricClass(
    #     prediction,
    #     y_test,
    # )
    #
    # return metric


def main():
    data_provider = DataProviders[args.data_provider_id]
    get_model, model_hyper_parameters, fit_hyper_parameters = MODELS[args.model_id]
    model = get_model(
        **model_hyper_parameters, **data_provider.get_data_format(),
    )

    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.environ['EXP_ID'] = \
        f'{args.model_id}_on_{args.data_provider_id}_{time_stamp}'
    os.mkdir(os.path.join(RESULT_DIR, os.environ.get('EXP_ID')))
    print('EXP_ID:', os.environ.get('EXP_ID'))

    flow(
        data_provider=data_provider,
        model=model,
        fit_hyper_parameters=fit_hyper_parameters,
    )


if __name__ == '__main__':
    main()
