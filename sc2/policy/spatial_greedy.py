import numpy as np
from sc2.policy import Policy

class SpatialGreedy(Policy):
    def __init__(self, spatial_extent=(64,64)):
        super().__init__()
        self.spatial_extent = spatial_extent
    def __call__(self, logits):
        if isinstance(logits,tuple) or \
           isinstance(logits,list): return logits
        argmax = logits.view(-1).argmax().item()
        xy = np.unravel_index(argmax, self.spatial_extent)
        return xy
