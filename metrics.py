import numpy as np
from medpy import metric as medmetric
from functools import partial

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


def cross_entropy(prob_pred, tar):
    if not ((tar == 0) | (tar == 1)).all():
        raise ValueError('Target data should be binary.')
    prob_pred = prob_pred[np.newaxis, :, :, :, :]
    tar = tar[np.newaxis, :, :, :, :]

    channel_num = tar.shape[1]
    temp = np.swapaxes(tar, 0, 1).reshape(channel_num, -1)
    weights = np.divide(
        1., np.mean(temp, axis=1),
        out=np.ones(channel_num),
        where=np.mean(temp, axis=1) != 0,
    )
    weights /= np.sum(weights)

    prob_pred = np.transpose(prob_pred, axes=[0, 4, 2, 3, 1])
    tar = np.transpose(tar, axes=[0, 4, 2, 3, 1])
    ce = tar * weights * np.log(prob_pred + epsilon)
    ce = -np.mean(np.sum(ce, axis=-1))
    return ce


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


class NTUMetric(MetricBase):
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


class BRATSMetric(MetricBase):
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


class StructSegHaNMetric(MetricBase):

    class_weights = {
        'left eye': 100,
        'right eye': 100,
        'left_lens': 50,
        'right lens': 50,
        'left optical nerve': 80,
        'right optical nerve': 80,
        'optical chiasma': 50,
        'pituitary': 80,
        'brain stem': 100,
        'left temporal lobes': 80,
        'right temporal lobes': 80,
        'spinal cord': 100,
        'left parotid gland': 50,
        'right parotid gland': 50,
        'left inner ear': 70,
        'right inner ear': 70,
        'left middle ear': 70,
        'right middle ear': 70,
        'left temporomandibular joint': 60,
        'right temporomandibular joint': 60,
        'left mandible': 100,
        'right mandible': 100,
    }

    def __init__(self, prob_pred, tar):
        super().__init__(prob_pred, tar)

        soft_dice_metrics = {
            f'soft_dice_{metric_name}': partial(self.one_class_metric_func, soft_dice, class_idx)
            for class_idx, metric_name in enumerate(self.class_weights.keys())
        }
        hard_dice_metrics = {
            f'hard_dice_{metric_name}': partial(self.one_class_metric_func, medmetric.dc, class_idx)
            for class_idx, metric_name in enumerate(self.class_weights.keys())
        }
        self.do_all_metrics = {
            'crossentropy': self.cross_entropy,
            **soft_dice_metrics,
            **hard_dice_metrics,
        }

    def one_class_metric_func(self, metric_func, class_idx):
        return volumewise_mean_score(metric_func, self.pred[:, class_idx], self.tar[:, class_idx])

    def cross_entropy(self):
        return volumewise_mean_score(cross_entropy, self.prob_pred, self.tar)

    def all_metrics(self, verbose=True):
        results = super(StructSegHaNMetric, self).all_metrics(verbose=False)
        total_class_weight = sum(self.class_weights.values())
        prefixs = ['soft_dice', 'hard_dice']
        accum_scores = {score_name: 0. for score_name in prefixs}

        for metric_name, score in results.items():
            for prefix in prefixs:
                if metric_name.startswith(prefix):
                    stripped_metric_name = metric_name[len(prefix) + 1:]
                    accum_scores[prefix] += \
                        score * self.class_weights[stripped_metric_name] / total_class_weight
                    break

        new_results = {
            **results,
            **{f'{prefix}_overall': accum_scores[prefix] for prefix in prefixs},
        }

        if verbose:
            for metric, result in new_results.items():
                print(f'{metric}: {result}')

        return new_results
