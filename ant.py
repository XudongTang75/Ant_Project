import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))  # Assuming normalized actions
        return action


# Return ants that align on X axis, and for y axis they are 5.0 apart.
def compute_env_offsets(num_envs, env_offset=(0.0, 5.0, 0.0), up_axis="Z"):
    # Determine the side length of the grid
    side_length = int(np.ceil(np.sqrt(num_envs)))
    env_offsets = []

    for i in range(num_envs):

        offset = np.zeros(3)
        offset[0] = 0.0
        offset[1] = i * env_offset[1]  # Y-axis position
        offset[2] = 0.0  # Fixed Z-axis position for all Ants

        env_offsets.append(offset)

    # Convert to a numpy array for easy manipulation
    env_offsets = np.array(env_offsets)

    # Calculate the center of the grid
    grid_center = np.mean(env_offsets, axis=0)

    # Line up the Ants on the Y axis
    env_offsets[:, 0] -= grid_center[0]
    env_offsets[:, 1] -= grid_center[1]

    return env_offsets


def anneal(frame_num, total_frames, start_temp, end_temp):
    """
    Gradually decreases the temperature from start_temp to end_temp over total_frames.

    :param frame_num: Current frame number.
    :param total_frames: Total number of frames over which to anneal.
    :param start_temp: Starting temperature.
    :param end_temp: Ending temperature.
    :return: Temperature at the current frame.
    """
    if frame_num < 0:
        frame_num = 0
    elif frame_num > total_frames:
        frame_num = total_frames

    ratio = frame_num / total_frames
    temperature = start_temp + (end_temp - start_temp) * ratio
    return temperature


@wp.kernel
def compute_reward(
    body_q: wp.array(dtype=wp.transform),  # The transformation of the entire body
    reward: wp.array(dtype=wp.float32),  # Output array for reward
):
    tid = wp.tid()  # Thread ID for each Ant
    com_0 = wp.transform_point(body_q[tid * 9], wp.vec3f(0.0, 0.0, 0.0))
    com_1 = wp.transform_point(body_q[tid * 9 + 1], wp.vec3f(0.0, 0.0, 0.0))
    com_2 = wp.transform_point(body_q[tid * 9 + 2], wp.vec3f(0.0, 0.0, 0.0))
    com_3 = wp.transform_point(body_q[tid * 9 + 3], wp.vec3f(0.0, 0.0, 0.0))
    com_4 = wp.transform_point(body_q[tid * 9 + 4], wp.vec3f(0.0, 0.0, 0.0))
    com_5 = wp.transform_point(body_q[tid * 9 + 5], wp.vec3f(0.0, 0.0, 0.0))
    com_6 = wp.transform_point(body_q[tid * 9 + 6], wp.vec3f(0.0, 0.0, 0.0))
    com_7 = wp.transform_point(body_q[tid * 9 + 7], wp.vec3f(0.0, 0.0, 0.0))
    com_8 = wp.transform_point(body_q[tid * 9 + 8], wp.vec3f(0.0, 0.0, 0.0))
    x = -(com_0 + com_1 + com_2 + com_3 + com_4 + com_5 + com_6 + com_7 + com_8) / 9.0
    reward[tid] = x[1]


class Example:
    def __init__(self, stage_path="ant.usd", num_envs=1, device: str = "cpu"):
        wp.set_device(device)
        articulation_builder = wp.sim.ModelBuilder()
        self.controller = PolicyNetwork(15 + 9 * 7, 8).to(device)
        self.optimizer = torch.optim.Adam(list(self.controller.parameters()), lr=1e-2)
        warp.sim.parse_mjcf(
            os.path.join(warp.examples.get_asset_directory(), "nv_ant.xml"),
            articulation_builder,
            xform=wp.transform([0.0, 0.55, 0.0], wp.quat_identity()),
            density=1000,
            armature=0.05,
            stiffness=0.1,
            damping=10,
            contact_ke=4.0e4,
            contact_kd=250.0,
            contact_kf=500.0,
            contact_mu=0.75,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder = wp.sim.ModelBuilder()
        self.sim_time = 0.0
        # duration of each simulation frame in seconds
        self.frame_dt = 1e-2

        # number of substeps per frame
        self.sim_substeps = 10
        # timestep for each substep
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.step_size = 1

        self.num_envs = num_envs
        self.offsets = compute_env_offsets(self.num_envs)

        self.target_origin = []
        for i in range(self.num_envs):
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(self.offsets[i], wp.quat_identity()),
            )

            builder.joint_q[-8:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            builder.joint_axis_mode = [wp.sim.JOINT_MODE_TARGET_POSITION] * len(
                builder.joint_axis_mode
            )
            # builder.joint_axis_mode = [wp.sim.JOINT_MODE_FORCE] * len(
            #     builder.joint_axis_mode
            # )
            # builder.joint_act[-8:] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

        np.set_printoptions(suppress=True)
        self.model = builder.finalize()
        self.model.ground = True
        self.model.joint_q.requires_grad = True
        self.model.body_q.requires_grad = True

        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

        self.states = [
            self.model.state(requires_grad=True) for _ in range(self.sim_substeps + 1)
        ]
        wp.sim.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, None, self.states[0]
        )
        self.control = self.model.control(requires_grad=True)

    def simulate(self, frame_num):
        self.optimizer.zero_grad()
        joint_signals = self.controller(
            torch.concatenate(
                [
                    wp.to_torch(self.states[0].joint_q).detach(),
                    wp.to_torch(self.states[0].body_q).detach().flatten(),
                ]
            ).unsqueeze(0)
        )[0]

        temp = anneal(
            frame_num=frame_num,
            total_frames=int(1000 / (self.sim_substeps * self.frame_dt)),
            start_temp=1.0,
            end_temp=0.01,
        )
        joint_signals = joint_signals + torch.randn_like(joint_signals) * 1
        self.control.joint_act = wp.from_torch(joint_signals, requires_grad=True)
        reward = wp.zeros((self.num_envs,), dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            for st in range(self.sim_substeps):
                self.states[st].clear_forces()
                wp.sim.collide(self.model, self.states[st])
                self.integrator.simulate(
                    self.model,
                    self.states[st],
                    self.states[st + 1],
                    self.sim_dt,
                    control=self.control,
                )
            wp.launch(
                compute_reward,
                dim=self.num_envs,
                inputs=[self.states[self.sim_substeps].body_q],
                outputs=[reward],
            )

        print(f"{frame_num} {temp} {reward} {joint_signals}")
        tape.backward(loss=reward)

        joint_signals.backward(gradient=joint_signals.grad)
        self.optimizer.step()
        tape.zero()

        self.states[0], self.states[-1] = self.states[-1], self.states[0]

    def step(self, frame_num):
        with wp.ScopedTimer("step", print=False):
            self.simulate(frame_num)
        self.sim_time += self.frame_dt * self.sim_substeps

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", print=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.states[0])
            self.renderer.end_frame()


# Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Override the default Warp device."
    )
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="ant.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument(
        "--num_frames", type=int, default=1000, help="Total number of frames."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Total number of simulated environments.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=args.stage_path, num_envs=args.num_envs, device=args.device
        )

        for i in range(args.num_frames):
            example.step(i)
            example.render()

        if example.renderer:
            example.renderer.save()
