import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

def compute_env_offsets(num_envs, env_offset=(5.0, 5.0, 0.0), up_axis="Z"):
    # Determine the side length of the grid
    side_length = int(np.ceil(np.sqrt(num_envs)))
    env_offsets = []

    for i in range(num_envs):
        # Calculate the X and Y positions based on the grid layout
        d0 = i // side_length
        d1 = i % side_length
        offset = np.zeros(3)
        offset[0] = d0 * env_offset[0]  # X-axis position based on row index
        offset[1] = d1 * env_offset[1]  # Y-axis position based on column index
        offset[2] = 0.0  # Fixed Z-axis position for all Ants

        env_offsets.append(offset)

    # Convert to a numpy array for easy manipulation
    env_offsets = np.array(env_offsets)

    # Calculate the center of the grid
    grid_center = np.mean(env_offsets, axis=0)
    
    # Subtract the grid center's X and Y values to center the grid around the environment origin
    env_offsets[:, 0] -= grid_center[0]
    env_offsets[:, 1] -= grid_center[1]

    return env_offsets

class Example:
    def __init__(self, stage_path="ant.usd", num_envs=8):
        articulation_builder = wp.sim.ModelBuilder()
        warp.sim.parse_mjcf(
            os.path.join(warp.examples.get_asset_directory(), "nv_ant.xml"),
            articulation_builder,
            xform=wp.transform([0.0, 0.55, 0.0], wp.quat_identity()),
            density=1000,
            armature=0.05,
            stiffness=0.0,
            damping=1,
            contact_ke=4.e+4,
            contact_kd=1.e+4,
            contact_kf=3.e+3,
            contact_mu=0.75,
            limit_ke=1.e+3,
            limit_kd=1.e+1,
        )
        builder = wp.sim.ModelBuilder()
        self.sim_time = 0.0
        fps = 100
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_envs = num_envs
        offsets = compute_env_offsets(self.num_envs)

        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

            builder.joint_q[-8:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            builder.joint_axis_mode = [wp.sim.JOINT_MODE_TARGET_POSITION] * len(builder.joint_axis_mode)
            builder.joint_act[-8:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]

        
        np.set_printoptions(suppress=True)

        self.model = builder.finalize()
        self.model.ground = True
        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0

        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)
        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None
        
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()




# Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="ant.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=8, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()