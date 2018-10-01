from functools import partial

from .toy_model import ToyModel


DEFAULT_TRAINING_PARAM = {
    'batch_size': 128,
    'epochs': 50,
    'verbose': 1,
    'shuffle': True,
}

MODELS = {
    'toy_model': (
        partial(
            ToyModel,
            {
                'num_units': 32,
            }),
        DEFAULT_TRAINING_PARAM,
    ),
}
