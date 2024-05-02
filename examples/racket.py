import numpy as np
import os
import trimesh
from PIL import Image
import warp as wp

import sapien
import warprigid
from warprigid.config import RigidConfig
from warprigid.system import RigidSystem
from warprigid.component import RigidComponent


# ffmpeg -framerate 50 -i ../output/racket/frames/step_%04d.png -c:v libx264 -crf 0 ../output/racket/racket.mp4

wp.init()

# init_pose = sapien.Pose([0, 0, 1], [1, 0, 0, 0])
# init_ang_vel = np.array([3, 0.1, 0.1], dtype=np.float32)
# init_ang_vel = np.array([0.1, 3, 0.1], dtype=np.float32)
# # init_ang_vel = np.array([0.1, 0.1, 3], dtype=np.float32)
init_lin_vel = np.array([0, 0, 0], dtype=np.float32)

render_path = "../output/racket/frames"
os.makedirs(render_path, exist_ok=True)

# obj_path = "../assets/box.obj"
obj_path = "../assets/racket.glb"
scale = 0.1

n_steps = 5000
render_every = 10

config = RigidConfig()
config.gravity = np.array([0, 0, 0], dtype=np.float32)
system = RigidSystem(config, "cuda:0")

scene = sapien.Scene()
scene.set_ambient_light([0.5, 0.5, 0.5])
scene.add_directional_light([1, 0, -1], [1, 1, 1], True)
scene.add_ground(0)
scene.add_system(system)

cam_entity = sapien.Entity()
cam = sapien.render.RenderCameraComponent(1600, 900)
cam.set_near(1e-3)
cam.set_far(1000)
cam.set_fovy(1.05)
cam_entity.add_component(cam)
cam_entity.name = "camera"
cam_entity.set_pose(
    sapien.Pose([-0.19722, -1.60262, 1.41675], [0.715731, -0.154562, 0.167584, 0.660118])
)
scene.add_entity(cam_entity)

def add_rigid(scene, obj_path, scale, init_pose, init_ang_vel, init_lin_vel):
    # V, F = igl.read_triangle_mesh(obj_path)
    mesh = trimesh.load_mesh(obj_path, enable_post_processing=True, process=False)
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            meshes = []
            for g in mesh.geometry.values():
                # print(g.volume)
                if g.volume > 1e-3:
                    meshes.append(trimesh.Trimesh(vertices=g.vertices, faces=g.faces))
            mesh = trimesh.util.concatenate(meshes)
    mesh.fill_holes()
    # print(mesh.is_watertight)
    V = mesh.vertices
    F = mesh.faces

    rigid = RigidComponent(V, F, 1e3, scale=scale)
    rigid.set_angular_velocity(init_ang_vel)
    rigid.set_linear_velocity(init_lin_vel)
    # print(rigid.vertices)
    # print("Mass:", rigid.mass)
    # print("CoM:", rigid.com)
    # print("Inertia:", rigid.inertia)

    rigid_render = sapien.render.RenderBodyComponent()
    rigid_render_shape = sapien.render.RenderShapeTriangleMesh(obj_path)
    rigid_render_shape.set_scale([scale, scale, scale])
    rigid_render.attach(rigid_render_shape)

    rigid_entity = sapien.Entity()
    rigid_entity.add_component(rigid)
    rigid_entity.add_component(rigid_render)
    rigid_entity.set_pose(init_pose)
    scene.add_entity(rigid_entity)

    return rigid

rigidx = add_rigid(scene, obj_path, scale, sapien.Pose([-1, 0, 1], [1, 0, 0, 0]), [3, 0.1, 0.1], init_lin_vel)
rigidy = add_rigid(scene, obj_path, scale, sapien.Pose([0, 0, 1], [1, 0, 0, 0]), [0.1, 3, 0.1], init_lin_vel)
rigidz = add_rigid(scene, obj_path, scale, sapien.Pose([1, 0, 1], [1, 0, 0, 0]), [0.1, 0.1, 3], init_lin_vel)

viewer = sapien.utils.Viewer()
viewer.set_scene(scene)
viewer.set_camera_pose(cam.get_entity_pose())
viewer.window.set_camera_parameters(1e-3, 1000, np.pi / 2)
viewer.paused = True


def save_render(step):
    cam.take_picture()
    rgba = cam.get_picture("Color")
    rgba = np.clip(rgba, 0, 1)[:, :, :3]
    rgba = Image.fromarray((rgba * 255).astype(np.uint8))
    rgba.save(os.path.join(render_path, f"step_{step:04d}.png"))


scene.update_render()
viewer.render()
# save_render(0)

for i in range(n_steps):
    system.step()
    if (i + 1) % render_every == 0:
        # rigid.update_render()
        for e in scene.get_entities():
            for c in e.get_components():
                if isinstance(c, RigidComponent):
                    c.update_render()
        scene.update_render()
        viewer.render()
        # save_render((i + 1) // render_every)


