import numpy as np


class GetClassWeights:

    decay_rate = 0.5

    @classmethod
    def __call__(cls, target, class_num=None, onehot=False):
        """
        :param target: shape (N, D, H, W) or (N, C, D, H, W)
        :return: weights for each class, shape (C,)
        """
        if class_num is None:
            assert onehot
            class_num = target.shape[1]

        if not onehot:
            unique, counts = np.unique(target, return_counts=True)
            cur_class_ratio = np.zeros([class_num], dtype=float)
            for class_idx, count in zip(unique, counts):
                cur_class_ratio[class_idx] = count / target.size
        else:
            cur_class_ratio = np.swapaxes(target, 0, 1).reshape(class_num, -1).mean(axis=-1)

        weights = np.divide(
            1., cur_class_ratio,
            out=np.ones(class_num),
            where=cur_class_ratio != 0,
        )
        return weights
