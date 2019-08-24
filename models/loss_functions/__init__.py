from .misc import (
    ce_minus_log_dice,
    weighted_cross_entropy,
)
from .dice import minus_dice


loss_function_hub = {
    'crossentropy-log(dice)': ce_minus_log_dice,
    'crossentropy': weighted_cross_entropy,
    'dice': minus_dice,
}
