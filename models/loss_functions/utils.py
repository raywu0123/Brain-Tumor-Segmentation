import numpy as np


class GetClassWeights:

    decay_rate = 0.5
    class_ratio = None

    @classmethod
    def __call__(cls, target):
        channel_num = target.shape[1]
        if cls.class_ratio is None:
            cls.class_ratio = np.zeros([channel_num], dtype=float)

        cur_class_ratio = np.swapaxes(target, 0, 1).reshape(channel_num, -1).mean(axis=-1)
        # cur_class_ratio /= np.sum(cur_class_ratio)

        cls.class_ratio = cls.class_ratio * cls.decay_rate + (1 - cls.decay_rate) * cur_class_ratio

        weights = np.divide(
            1., cls.class_ratio,
            out=np.ones(channel_num),
            where=cls.class_ratio != 0,
        )
        return weights
