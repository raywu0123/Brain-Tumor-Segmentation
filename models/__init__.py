from functools import partial

from .toy_model import ToyModel


DEFAULT_TRAINING_PARAM = {
    'batch_size': 20,
    'epoch_num': 6000,
    'verbose_epoch_num': 10,
}

MODELS = {
    'toy_model': (
        partial(
            ToyModel,
        ),
        DEFAULT_TRAINING_PARAM,
    ),
}
