from functools import partial

from .two_dim import TwoDimBatchSampler
from .three_dim import ThreeDimBatchSampler
from .uniform_patch3d import UniformPatch3DBatchSampler
from .center_patch3d import CenterPatch3DBatchSampler
from .two_and_half_dim import TwoAndHalfDimBatchSampler


BatchSamplerHub = {
    'two_dim': TwoDimBatchSampler,
    'three_dim': ThreeDimBatchSampler,
    'uniform_patch3d': UniformPatch3DBatchSampler,
    'center_patch3d': CenterPatch3DBatchSampler,
    'two_dim_depth2': partial(TwoAndHalfDimBatchSampler, depth=2),
    'two_dim_depth3': partial(TwoAndHalfDimBatchSampler, depth=3),
    'two_dim_depth4': partial(TwoAndHalfDimBatchSampler, depth=4),
    'two_dim_depth5': partial(TwoAndHalfDimBatchSampler, depth=5),
}
