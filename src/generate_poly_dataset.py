import numpy as np
import pyvista as pv
import random as rd
import argparse
import os
import math
from typing import Tuple, Iterator

MIN_R = 2.0
MAX_R = 6.0
MAX_DEG_PER_STEP = 90.0

SHAPE_REGISTRY = {
    'tetrahedron': 0,
    'cube': 1,
    'octahedron': 2,
    'dodecahedron': 3,
    'icosahedron': 4
}
SHAPE_NAMES = list(SHAPE_REGISTRY.keys())


def get_mesh(kind: str) -> pv.PolyData:
    return pv.PlatonicSolid(kind, radius=0.4, center=(0, 0, 0))


def closed_loop_trajectory(
    length: int, initial_state: Tuple[float, float, float]
) -> Iterator[Tuple[float, float, float]]:
    velocities = []
    
    sd_0, sy_0, sx_0 = initial_state
    sd_i, sy_i, sx_i = initial_state
    vd_i, vy_i, vx_i = 0.0, 0.0, 0.0
    
    if length % 2 != 0:
        length += 1

    for i in range(length):
        if i == length - 1 or i == length - 2:
            vd_i = -(sd_i - sd_0) / 2
            vy_i = -(sy_i - sy_0) / 2
            vx_i = -(sx_i - sx_0) / 2
        else:
            if i % 2 == 0:
                vd_i = rd.uniform(-1 - sd_i, 1 - sd_i) / 2
                vx_i = rd.uniform(-1, 1) / 2
                vy_i = rd.uniform(-1, 1) / 2
            
            sd_i += vd_i
            sx_i += vx_i
            sy_i += vy_i

        velocities.append((vd_i, vy_i, vx_i))

    for vel in velocities:
        yield vel


def apply_action_and_get_image(
    plotter: pv.Plotter,
    actor: pv.Actor,
    current_state: Tuple[float, float, float],
    action_vel: Tuple[float, float, float],
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    
    d0, ry0, rx0 = current_state
    vd, vy, vx = action_vel

    d_final = np.clip(d0 + vd, -1, 1)
    ry_final = ry0 + (vy * MAX_DEG_PER_STEP)
    rx_final = rx0 + (vx * MAX_DEG_PER_STEP)

    # Clamping zoom based on arm length restriction
    r = ((MAX_R + MIN_R) / 2) + (d_final * (MAX_R - MIN_R) / 2)
    
    plotter.camera.position = (0, 0, r)
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = (0, 1, 0)

    actor.orientation = [rx_final, ry_final, 0]

    plotter.render()
    img = plotter.screenshot(transparent_background=True, return_img=True)

    return img, (d_final, ry_final, rx_final)


def generate_raw_dataset(
    output_dir: str,
    num_trajectories: int,
    shard_size: int,
    trajectory_length: int,
    resolution: int,
    shape_arg: str,
    monochromatic: bool = False
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pl = pv.Plotter(window_size=[resolution, resolution], off_screen=True)
    pl.set_background("black")
    pl.hide_axes()

    total_shards = math.ceil(num_trajectories / shard_size)
    print(f"Generating {num_trajectories} trajectories ({shape_arg}) across {total_shards} shards.")

    traj_global_count = 0

    for shard_idx in range(total_shards):
        current_shard_size = min(shard_size, num_trajectories - traj_global_count)
        
        shard_images = []
        shard_actions = []
        shard_states = []
        shard_shape_ids = []

        if shape_arg == 'mixed':
            assigned_shapes = [SHAPE_NAMES[i % len(SHAPE_NAMES)] for i in range(current_shard_size)]
            rd.shuffle(assigned_shapes)
        else:
            assigned_shapes = [shape_arg] * current_shard_size

        for i in range(current_shard_size):
            traj_imgs = []
            traj_vels = []
            traj_states = []
            
            current_shape_name = assigned_shapes[i]
            current_shape_id = SHAPE_REGISTRY[current_shape_name]
            
            pl.clear_actors() 
            mesh = get_mesh(current_shape_name)
            
            if monochromatic:
                actor = pl.add_mesh(mesh, color="white", show_edges=True, line_width=2)
            else:
                if 'face_ids' not in mesh.cell_data:
                    mesh.cell_data['face_ids'] = np.arange(mesh.n_cells)
                
                actor = pl.add_mesh(
                    mesh,
                    show_edges=True,
                    line_width=2,
                    cmap="tab20",
                    scalars='face_ids',
                    preference="cell",
                    show_scalar_bar=False,
                )

            start_d = rd.uniform(-1, 1)
            start_ry = rd.uniform(0, 360)
            start_rx = rd.uniform(0, 360)
            state = (start_d, start_ry, start_rx)

            img, state = apply_action_and_get_image(pl, actor, state, (0.0, 0.0, 0.0))
            traj_imgs.append(img)
            traj_states.append(state)
            
            velocities = list(closed_loop_trajectory(trajectory_length, state))
            
            for vel in velocities:
                img, state = apply_action_and_get_image(pl, actor, state, vel)
                traj_imgs.append(img)
                traj_states.append(state)
                traj_vels.append(vel)

            shard_images.append(np.array(traj_imgs, dtype=np.uint8))
            shard_actions.append(np.array(traj_vels, dtype=np.float32))
            shard_states.append(np.array(traj_states, dtype=np.float32))
            shard_shape_ids.append(current_shape_id)
            
            traj_global_count += 1
            print(f"  [Shard {shard_idx}] {i+1}/{current_shard_size}", end="\r")

        save_name = f"shard{shard_idx:03d}.npz"
        save_path = os.path.join(output_dir, save_name)
        
        np.savez_compressed(
            save_path,
            images=np.array(shard_images),
            actions=np.array(shard_actions),
            states=np.array(shard_states),
            shape_ids=np.array(shard_shape_ids)
        )
        print(f"\n  Saved: {save_path}")

    pl.close()
    print(f"Done. {traj_global_count} trajectories generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trajs", type=int, default=100)
    parser.add_argument("--shard_size", type=int, default=50)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./raw_data")
    parser.add_argument("--monochromatic", action="store_true")
    parser.add_argument("--shape", type=str, default="dodecahedron", choices=SHAPE_NAMES + ['mixed'])

    args = parser.parse_args()

    generate_raw_dataset(
        args.output_dir,
        args.num_trajs, 
        args.shard_size,
        args.length, 
        args.resolution,
        args.shape,
        args.monochromatic
    )