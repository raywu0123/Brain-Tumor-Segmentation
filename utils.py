import numpy as np


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
        }

    def accuracy(self):
        acc = np.mean(
            np.logical_and(
                self.pred,
                self.tar
            )
        )
        return acc

    def all_metrics(self):
        results = {metric: metric_func() for (metric, metric_func) in self.do_all_metrics.items()}
        for metric, result in results.items():
            print(f'{metric}: {result}')
        return results
