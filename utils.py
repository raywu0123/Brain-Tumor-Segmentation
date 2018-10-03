import numpy as np
from medpy import metric as medmetric


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
        self.pred = pred
        self.tar = tar
        self.do_all_metrics = {
            'accuracy': self.accuracy,
            'dice': self.dice,
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

    def dice(self):
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
