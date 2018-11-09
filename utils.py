import numpy as np
from medpy import metric as medmetric

from data.utils import to_one_hot_label


def parse_exp_id(exp_id_string):
    splits = exp_id_string.split('_on_')
    model_id = splits[0]
    data_id = '_'.join(splits[-1].split('_')[:-1])
    time_stamp = splits[-1].split('_')[-1]
    return model_id, data_id, time_stamp


class MetricClass:
    def __init__(
            self,
            pred,
            tar,
    ):
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
        # Strip background
        pred = pred[:, 1:]
        tar = tar[:, 1:]

        self.prob_pred = pred
        self.pred = self.hard_max(pred)
        self.tar = tar
        self.do_all_metrics = {
            'accuracy': self.accuracy,
            'dice-score': self.dice_score,
            'dice-loss': self.dice_loss,
            'sensitivity': self.sensitivity,
            'precision': self.precision,
            # 'assd': self.assd,
        }

    def accuracy(self):
        acc = 1 - np.mean(
            np.logical_xor(
                self.pred,
                self.tar
            )
        )
        return acc

    def dice_loss(self):
        intersection = self.tar * self.prob_pred
        dice_loss = (2 * np.sum(intersection) + 1) / (np.sum(self.prob_pred) + np.sum(self.tar) + 1)
        return dice_loss

    def dice_score(self):
        return medmetric.dc(self.pred, self.tar)

    def sensitivity(self):
        return medmetric.sensitivity(self.pred, self.tar)

    def precision(self):
        return medmetric.precision(self.pred, self.tar)

    def hausdorff_distance(self):
        return medmetric.hd(self.pred, self.tar)

    def assd(self):
        return medmetric.assd(self.pred, self.tar)

    def all_metrics(self):
        results = {metric: metric_func() for (metric, metric_func) in self.do_all_metrics.items()}
        for metric, result in results.items():
            print(f'{metric}: {result}')
        return results

    @staticmethod
    def hard_max(x):
        index_x = np.argmax(x, axis=1)
        categorical_x = to_one_hot_label(index_x, class_num=x.shape[1])
        categorical_x = np.moveaxis(categorical_x, 0, 1)
        return categorical_x
