import numpy as np
from medpy import metric as medmetric

from data.utils import to_one_hot_label
from utils import epsilon


def hard_max(x):
    index_x = np.argmax(x, axis=1)
    categorical_x = to_one_hot_label(index_x, class_num=x.shape[1])
    categorical_x = np.moveaxis(categorical_x, 0, 1)
    return categorical_x


def soft_dice(prob_pred, tar):
    if not ((tar == 0) | (tar == 1)).all():
        raise ValueError('Target data should be binary.')
    intersection = tar * prob_pred
    dice_loss = (2 * np.sum(intersection) + epsilon) \
        / (np.sum(prob_pred ** 2) + np.sum(tar ** 2) + epsilon)
    return dice_loss


def volumewise_mean_score(score_fn, pred_batch, tar_batch):
    return np.mean([score_fn(pred, tar) for pred, tar in zip(pred_batch, tar_batch)])


class MetricBase:
    def __init__(self, pred, tar):
        if pred.shape != tar.shape:
            raise ValueError(
                f"pred.shape should be equal to tar.shape, "
                f"got pred = {pred.shape} and tar = {tar.shape}",
            )
        if pred.shape[1] < 2:
            raise ValueError(
                f'pred.shape[1] (class_num) should be greater than 1, '
                f'got class_num = {pred.shape[1]}'
            )
        if pred.ndim != 5:
            raise ValueError(
                'Input shape of Metric-Class should be (N, C, D, H, W), '
                f'got {pred.shape} instead.'
            )
        # Strip background
        self.prob_pred = pred[:, 1:]
        self.pred = hard_max(pred)[:, 1:]
        self.tar = tar[:, 1:]
        self.do_all_metrics = {}

    def all_metrics(self, verbose=True):
        results = {metric: metric_func() for (metric, metric_func) in self.do_all_metrics.items()}
        if verbose:
            for metric, result in results.items():
                print(f'{metric}: {result}')
        return results


class MetricClass(MetricBase):
    def __init__(
            self,
            pred,
            tar,
    ):
        super().__init__(pred, tar)
        self.do_all_metrics = {
            'soft_dice': self.soft_dice,
            'hard_dice': self.hard_dice,
            'sensitivity': self.sensitivity,
            'precision': self.precision,
        }

    def soft_dice(self):
        return volumewise_mean_score(soft_dice, self.prob_pred, self.tar)

    def hard_dice(self):
        return volumewise_mean_score(medmetric.dc, self.pred, self.tar)

    def sensitivity(self):
        return volumewise_mean_score(medmetric.sensitivity, self.pred, self.tar)

    def precision(self):
        return volumewise_mean_score(medmetric.precision, self.pred, self.tar)


class BRATSMetricClass(MetricBase):
    def __init__(self, prob_pred, tar):
        super().__init__(prob_pred, tar)
        self.prob_pred_complete = \
            prob_pred[:, 1] + prob_pred[:, 2] + prob_pred[:, 3] + prob_pred[:, 4]
        self.prob_pred_core = prob_pred[:, 1] + prob_pred[:, 3] + prob_pred[:, 4]
        self.prob_pred_enhancing = prob_pred[:, 4]

        pred = hard_max(prob_pred)
        self.pred_complete = pred[:, 1] + pred[:, 2] + pred[:, 3] + pred[:, 4]
        self.pred_core = pred[:, 1] + pred[:, 3] + pred[:, 4]
        self.pred_enhancing = pred[:, 4]

        self.tar_complete = tar[:, 1] + tar[:, 2] + tar[:, 3] + tar[:, 4]
        self.tar_core = tar[:, 1] + tar[:, 3] + tar[:, 4]
        self.tar_enhancing = tar[:, 4]

        self.do_all_metrics = {
            'soft_dice_complete': self.soft_dice_complete,
            'soft_dice_core': self.soft_dice_core,
            'soft_dice_enhancing': self.soft_dice_enhancing,

            'hard_dice_complete': self.hard_dice_complete,
            'hard_dice_core': self.hard_dice_core,
            'hard_dice_enhancing': self.hard_dice_enhancing,

            'precision_complete': self.precision_complete,
            'precision_core': self.precision_core,
            'precision_enhancing': self.precision_enhancing,

            'sensitivity_complete': self.sensitivity_complete,
            'sensitivity_core': self.sensitivity_core,
            'sensitivity_enhancing': self.sensitivity_enhancing,
        }

    def soft_dice_complete(self):
        return volumewise_mean_score(soft_dice, self.prob_pred_complete, self.tar_complete)

    def soft_dice_core(self):
        return volumewise_mean_score(soft_dice, self.prob_pred_core, self.tar_core)

    def soft_dice_enhancing(self):
        return volumewise_mean_score(soft_dice, self.prob_pred_enhancing, self.tar_enhancing)

    def hard_dice_complete(self):
        return volumewise_mean_score(medmetric.dc, self.pred_complete, self.tar_complete)

    def hard_dice_core(self):
        return volumewise_mean_score(medmetric.dc, self.pred_core, self.tar_core)

    def hard_dice_enhancing(self):
        return volumewise_mean_score(medmetric.dc, self.pred_enhancing, self.tar_enhancing)

    def precision_complete(self):
        return volumewise_mean_score(medmetric.precision, self.pred_complete, self.tar_complete)

    def precision_core(self):
        return volumewise_mean_score(medmetric.precision, self.pred_core, self.tar_core)

    def precision_enhancing(self):
        return volumewise_mean_score(medmetric.precision, self.pred_enhancing, self.tar_enhancing)

    def sensitivity_complete(self):
        return volumewise_mean_score(medmetric.sensitivity, self.pred_complete, self.tar_complete)

    def sensitivity_core(self):
        return volumewise_mean_score(medmetric.sensitivity, self.pred_core, self.tar_core)

    def sensitivity_enhancing(self):
        return volumewise_mean_score(medmetric.sensitivity, self.pred_enhancing, self.tar_enhancing)
