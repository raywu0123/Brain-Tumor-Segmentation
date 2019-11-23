import numpy as np
from medpy import metric as medmetric
from functools import partial

from utils import to_one_hot_label
from utils import epsilon
from models.loss_functions.utils import GetClassWeights


def hard_max(x):
    index_x = np.argmax(x, axis=1)
    categorical_x = to_one_hot_label(index_x, class_num=x.shape[1])
    return categorical_x


def soft_dice(prob_pred, tar):
    if not ((tar == 0) | (tar == 1)).all():
        raise ValueError('Target data should be binary.')
    intersection = tar * prob_pred
    intersection = np.sum(intersection)
    m1 = np.sum(prob_pred ** 2)
    m2 = np.sum(tar ** 2)
    dice_loss = (2 * intersection + epsilon) / (m1 + m2 + epsilon)
    return dice_loss


def cross_entropy(prob_pred, tar_ids):
    class_num = prob_pred.shape[1]
    weights = GetClassWeights()(tar_ids, class_num=class_num)
    weights /= class_num

    selected_pred = np.take_along_axis(
        prob_pred,  # (N, C, ...)
        tar_ids[:, np.newaxis],  # (N, 1, ...)
        axis=1,
    )  # (N, 1, ...)
    selected_weights = weights[tar_ids]  # (N, ...)
    ce = -selected_weights * np.log(np.squeeze(selected_pred, axis=1) + epsilon)  # (N, ...)
    ce = np.mean(ce)
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
        self.do_all_metrics = {}

    def all_metrics(self, verbose=True):
        results = {metric: metric_func() for (metric, metric_func) in self.do_all_metrics.items()}
        if verbose:
            for metric, result in results.items():
                print(f'{metric}: {result:.2f}')
        return results


class ClasswiseMetric(MetricBase):

    def __init__(self, pred, tar):
        self.class_num = pred.shape[1]  # class_num includes background
        self.tar_ids = tar
        tar = to_one_hot_label(tar, self.class_num)
        super().__init__(pred, tar)

        self.prob_pred = pred
        self.pred = hard_max(pred)
        self.tar = tar

        self.metrics = {
            'soft_dice': (soft_dice, self.prob_pred),
            'hard_dice': (medmetric.dc, self.pred),
            'sensitivity': (medmetric.sensitivity, self.pred),
            'precision': (medmetric.precision, self.pred),
        }

        self.do_all_metrics = {
            **{
                f'{metric_name}_class{i}': partial(
                    volumewise_mean_score,
                    score_fn=metric_fn,
                    pred_batch=p[:, i],
                    tar_batch=self.tar[:, i],
                )
                for metric_name, (metric_fn, p) in self.metrics.items()
                for i in range(1, self.class_num)
            },
            'crossentropy': partial(cross_entropy, self.prob_pred, self.tar_ids),
        }

    def all_metrics(self, verbose=True):
        results = super(ClasswiseMetric, self).all_metrics(verbose=False)
        accum_scores = {score_name: 0. for score_name in self.metrics.keys()}

        for metric_name, score in results.items():
            for prefix in self.metrics.keys():
                if metric_name.startswith(prefix):
                    accum_scores[prefix] += score / (self.class_num - 1)  # minus 1 for background
                    break

        new_results = {
            **results,
            **{f'{prefix}_overall': accum_scores[prefix] for prefix in self.metrics.keys()},
        }

        if verbose:
            for metric, result in new_results.items():
                print(f'{metric}: {result:.2f}')
        return new_results


class BRATSMetric(MetricBase):
    def __init__(self, prob_pred, tar):
        tar = to_one_hot_label(tar, prob_pred.shape[1])
        super().__init__(prob_pred, tar)

        self.pred_complete = \
            prob_pred[:, 1] + prob_pred[:, 2] + prob_pred[:, 3] + prob_pred[:, 4]
        self.pred_core = prob_pred[:, 1] + prob_pred[:, 3] + prob_pred[:, 4]
        self.pred_enhancing = prob_pred[:, 4]

        self.tar_complete = tar[:, 1] + tar[:, 2] + tar[:, 3] + tar[:, 4]
        self.tar_core = tar[:, 1] + tar[:, 3] + tar[:, 4]
        self.tar_enhancing = tar[:, 4]

        self.metrics = {
            'soft_dice': soft_dice,
            'hard_dice': medmetric.dc,
            'sensitivity': medmetric.sensitivity,
            'precision': medmetric.precision,
        }
        self.modes = {
            'complete': (self.pred_complete, self.tar_complete),
            'core': (self.pred_core, self.tar_core),
            'enhancing': (self.pred_enhancing, self.tar_enhancing),
        }
        self.do_all_metrics = {
            f'{metric_name}_{mode}': partial(
                volumewise_mean_score,
                score_fn=metric_fn,
                pred_batch=p,
                tar_batch=t,
            )
            for metric_name, metric_fn in self.metrics.items()
            for mode, (p, t) in self.modes.items()
        }


class StructSegHaNMetric(MetricBase):

    class_weights = {
        'left eye': 100,
        'right eye': 100,
        'left lens': 50,
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

    def __init__(self, pred, tar):
        self.tar_ids = tar
        tar = to_one_hot_label(tar, pred.shape[1])
        super().__init__(pred, tar)

        self.prob_pred = pred
        self.pred = hard_max(pred)
        self.tar = tar

        self.metrics = {
            'soft_dice': (soft_dice, self.prob_pred),
            'hard_dice': (medmetric.dc, self.pred),
        }
        self.do_all_metrics = {
            **{
                f'{metric_name}_{organ_name}': partial(
                    volumewise_mean_score,
                    metric_fn,
                    p[:, class_idx],
                    self.tar[:, class_idx],
                )
                for metric_name, (metric_fn, p) in self.metrics.items()
                for class_idx, organ_name in enumerate(self.class_weights.keys(), 1)
            },
            'crossentropy': partial(
                cross_entropy, self.prob_pred, self.tar_ids,
            ),
        }

    def all_metrics(self, verbose=True):
        results = super(StructSegHaNMetric, self).all_metrics(verbose=False)
        total_class_weight = sum(self.class_weights.values())
        accum_scores = {score_name: 0. for score_name in self.metrics.keys()}

        for metric_name, score in results.items():
            for prefix in self.metrics.keys():
                if metric_name.startswith(prefix):
                    stripped_metric_name = metric_name[len(prefix) + 1:]
                    accum_scores[prefix] += \
                        score * self.class_weights[stripped_metric_name] / total_class_weight
                    break

        new_results = {
            **results,
            **{f'{prefix}_overall': accum_scores[prefix] for prefix in self.metrics.keys()},
        }

        if verbose:
            for metric, result in new_results.items():
                print(f'{metric}: {result:.2f}')
        return new_results
