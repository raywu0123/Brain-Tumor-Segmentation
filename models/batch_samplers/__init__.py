from functools import partial

from .two_dim import TwoDimBatchSampler
from .two_dim_aug import TwoDimAugBatchSampler
from .three_dim import ThreeDimBatchSampler
from .uniform_patch3d import UniformPatch3DBatchSampler
from .center_patch3d import CenterPatch3DBatchSampler


BatchSamplerHub = {
    'two_dim': TwoDimBatchSampler,
    'two_dim_aug': TwoDimAugBatchSampler,
    'three_dim': ThreeDimBatchSampler,
    'uniform_patch3d': UniformPatch3DBatchSampler,
    'center_patch_96': partial(CenterPatch3DBatchSampler, patch_size=(96, 96, 96)),
    'center_patch_structseg': partial(CenterPatch3DBatchSampler, patch_size=(152, 128, 128)),
}
