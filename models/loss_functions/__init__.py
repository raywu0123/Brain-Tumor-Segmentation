from functools import partial

from .misc import (
    ce_minus_log_dice,
    weighted_cross_entropy,
    minus_dice,
)


loss_function_hub = {
    'crossentropy-log[my_dice]': partial(ce_minus_log_dice, dice_type='my'),
    'crossentropy-log[generalized_dice]': partial(ce_minus_log_dice, dice_type='generalized'),
    'crossentropy-log[naive_dice]': partial(ce_minus_log_dice, dice_type='naive'),
    'my_dice': partial(minus_dice, dice_type='my'),
    'generalized_dice': partial(minus_dice, dice_type='generalized'),
    'naive_dice': partial(minus_dice, dice_type='naive'),
    'crossentropy': weighted_cross_entropy,
}
