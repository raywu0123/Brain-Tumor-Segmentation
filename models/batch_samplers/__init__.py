from .two_dim import TwoDimBatchSampler
from .three_dim import ThreeDimBatchSampler
from .uniform_patch3d import UniformPatch3DBatchSampler
from .center_patch3d import CenterPatch3DBatchSampler


BatchSamplerHub = {
    'two_dim': TwoDimBatchSampler,
    'three_dim': ThreeDimBatchSampler,
    'uniform_patch3d': UniformPatch3DBatchSampler,
    'center_patch3d': CenterPatch3DBatchSampler,
}
