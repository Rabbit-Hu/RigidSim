import typing
from typing import Union, List

import numpy as np
import scipy.sparse as sp
import warp as wp
import sapien
from sapien.render import RenderCudaMeshComponent

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import RigidSystem


class RigidComponent(sapien.Component):
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        density: float,
        scale: float = 1.0,
        mass: float = None,
        com: np.ndarray = None,
        inertia: np.ndarray = None,
    ):
        super().__init__()

        self.id_in_sys = None

        self.vertices = vertices * scale
        self.faces = faces
        self.density = density
        self._init_mass_properties(mass, com, inertia)
        self.linear_velocity = np.zeros(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)

    def _init_mass_properties(self, mass=None, com=None, inertia=None):
        if com:
            self.com = com
        else:
            V = self.vertices
            F = V[self.faces]  # (N, 3 (triangle), 3 (dim))
            tet_com = F.sum(axis=1) / 4.0  # (N, 3)
            volumes = np.linalg.det(F) / 6.0  # (N,)
            self.com = (volumes[:, None] * tet_com).sum(axis=0) / volumes.sum()
        self.vertices -= self.com

        if mass: 
            self.mass = mass
        else:
            V = self.vertices
            F = V[self.faces]  # (N, 3 (triangle), 3 (dim))
            volumes = np.linalg.det(F) / 6.0  # (N,)
            self.mass = volumes.sum() * self.density

        if inertia:
            self.inertia = inertia
        else:
            self.inertia = np.zeros((3, 3))

            V = self.vertices
            F = V[self.faces]  # (N, 3 (triangle), 3 (dim))
            volumes = np.linalg.det(F) / 6.0  # (N,)
            
            x_col = F[:, :, 0]  # (N, 3)
            y_col = F[:, :, 1]
            z_col = F[:, :, 2]

            def I_ab(a, b):
                return np.einsum("ij,ik->i", a, b) + np.einsum("ij,ij->i", a, b)

            I_xx = I_ab(x_col, x_col)
            I_yy = I_ab(y_col, y_col)
            I_zz = I_ab(z_col, z_col)
            I_xy = I_ab(x_col, y_col)
            I_yz = I_ab(y_col, z_col)
            I_zx = I_ab(z_col, x_col)

            I = np.empty((len(F), 3, 3), dtype=np.float32)

            I[:, 0, 0] = I_yy + I_zz
            I[:, 1, 1] = I_zz + I_xx
            I[:, 2, 2] = I_xx + I_yy
            I[:, 0, 1] = -I_xy
            I[:, 1, 0] = -I_xy
            I[:, 1, 2] = -I_yz
            I[:, 2, 1] = -I_yz
            I[:, 0, 2] = -I_zx
            I[:, 2, 0] = -I_zx

            I = I * volumes[:, None, None] * self.density / 20.0  # (N, 3, 3)

            self.inertia = I.sum(axis=0)

    def is_in_scene(self) -> bool:
        return self.entity and self.entity.scene

    def set_linear_velocity(self, v: np.ndarray):
        self.linear_velocity = v
        assert not self.is_in_scene(), "Setting velocity after adding to scene is not supported yet"

    def set_angular_velocity(self, w: np.ndarray):
        self.angular_velocity = w
        assert not self.is_in_scene(), "Setting velocity after adding to scene is not supported yet"

    def on_add_to_scene(self, scene: sapien.Scene):
        s: RigidSystem = scene.get_system("warprigid")
        s.register_rigid_component(self)

    def update_render(self):
        assert self.is_in_scene(), "Component must be added to scene before updating render"

        b_i = self.id_in_sys
        s: RigidSystem = self.entity.scene.get_system("warprigid")
        q = s.rigid_q.numpy()[b_i]
        pos = q[:3]
        quat = q[3:]
        # print(f"before: {pos}, {quat}; com: {self.com}")
        pos -= wp.quat_rotate(wp.quat(quat), wp.vec3(self.com))
        # print(f"after: {pos}, {quat}")
        quat = np.concatenate([quat[3:], quat[:3]])
        self.entity.set_pose(sapien.Pose(pos, quat))
        # print(f"entity.pose: {entity.pose}")