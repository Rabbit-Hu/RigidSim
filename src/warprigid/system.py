import typing
from typing import Union

import numpy as np
import warp as wp
import sapien

from .kernels.step_kernels import *

from .config import RigidConfig
from .component import RigidComponent
from .utils import wp_slice


class RigidSystem(sapien.System):
    def __init__(self, config: RigidConfig, device="cuda:0"):
        super().__init__()

        self.config = config
        self.device = device
        self.name = "warprigid"

        self.gravity = wp.vec3(*config.gravity)

        MR = self.config.max_rigid

        with wp.ScopedDevice(self.device):
            self.rigid_q = wp.zeros(MR, dtype=wp.transform)  # (quat, pos)
            self.rigid_qd = wp.zeros(
                MR, dtype=wp.spatial_vector
            )  # (omega, vel) in body frame
            self.rigid_mass = wp.zeros(MR, dtype=wp.float32)
            self.rigid_com = wp.zeros(MR, dtype=wp.vec3)
            self.rigid_inertia = wp.zeros(MR, dtype=wp.mat33)

        self.n_rigid = 0

    def get_name(self) -> str:
        return self.name

    def register_rigid_component(self, c: RigidComponent):
        assert (
            self.n_rigid < self.config.max_rigid
        ), f"Too many rigid bodies (max {self.config.max_rigid}, got {self.n_rigid + 1})"
        rid = self.n_rigid
        c.id_in_sys = rid
        self.n_rigid += 1

        with wp.ScopedDevice(self.device):
            T_sw = c.entity.pose.to_transformation_matrix()
            wp_slice(self.rigid_q, rid, rid + 1).assign(
                wp.transform(
                    c.entity.pose.p + T_sw[:3, :3] @ c.com,
                    np.concatenate((c.entity.pose.q[1:], c.entity.pose.q[:1])),
                )
            )
            wp_slice(self.rigid_qd, rid, rid + 1).assign(
                wp.spatial_vector(
                    np.concatenate((c.angular_velocity, c.linear_velocity))
                )
            )
            wp_slice(self.rigid_mass, rid, rid + 1).fill_(c.mass)
            wp_slice(self.rigid_com, rid, rid + 1).assign(c.com)
            wp_slice(self.rigid_inertia, rid, rid + 1).assign(c.inertia)

    def step(self):
        wp.launch(
            kernel=kinematic_step_explicit,
            dim=self.n_rigid,
            inputs=[
                self.rigid_q,
                self.rigid_qd,
                self.rigid_mass,
                self.rigid_com,
                self.rigid_inertia,
                self.config.time_step,
                self.gravity,
            ]
        )
    