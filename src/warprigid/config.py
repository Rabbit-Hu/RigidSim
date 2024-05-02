import numpy as np
import warp as wp


class RigidConfig:
    def __init__(self):
        self.max_rigid = 1 << 20

        self.time_step = 2e-3

        self.gravity = np.array([0, 0, -9.8], dtype=np.float32)
    