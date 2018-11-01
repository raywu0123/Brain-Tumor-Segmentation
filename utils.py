import numpy as np
from medpy import metric as medmetric


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
                f"pred.shape should be eqaul to tar.shape, "
                f"got pred = {pred.shape} and tar = {tar.shape}",
            )
        self.prob_pred = pred
        self.pred = (pred > 0.5).astype(int)
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
