import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

@wp.kernel
def compute_endeffector_position(
    body_q: wp.array(dtype=wp.transform),   # The transformation of the entire body
    # FIXME
    # MIGHT WANT TO CHANGE!!!
    num_links: int,                         # Number of links (joints) in each leg
    ee_link_indices: wp.array(dtype=int),   # Indices of the last link (feet) in the chain for each leg
    ee_link_offsets: wp.vec3,  # Offsets for each end-effector relative to its leg's last joint
    ee_pos: wp.array(dtype=wp.vec3)         # Output array for end-effector positions (all 4 feet per Ant)
):
    tid = wp.tid()  # Thread ID for each Ant
    # Compute positions for all 4 end-effectors (feet) for this Ant

    # FIXME
    # I think this is very wrong
    # Need to select all 4 ankle in body Q
    # Should be [tid*num_link + index_of_ankle 1/2/3/4]
    for i in range(4):  # Assuming 4 legs for the Ant
        ee_pos[tid * 4 + i] = wp.transform_point(
            body_q[tid * num_links + ee_link_indices[i]],  # Get the transformation of the last link in each leg
            ee_link_offsets  # Apply the offset for each foot
        )

class Ant:
    def __init__(self, stage_path="ant.usd", num_envs=10):
        self.num_envs = num_envs
        fps = 60
        self.frame_dt = 1.0 / fps
        self.render_time = 0.0
        self.step_size = 0.1


        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        articulation_builder = wp.sim.ModelBuilder()

        warp.sim.parse_mjcf(
            os.path.join(warp.examples.get_asset_directory(), "nv_ant.xml"),
            articulation_builder,
            xform=wp.transform([0.0, 0.55, 0.0], wp.quat_identity()),
            density=1000,
            armature=0.05,
            stiffness=0.0,
            damping=10,
            contact_ke=4.e+4,
            contact_kd=1.e+4,
            contact_kf=3.e+3,
            contact_mu=1,
            limit_ke=1.e+3,
            limit_kd=1.e+1,
        )

        builder = wp.sim.ModelBuilder()
        self.num_links = len(articulation_builder.joint_type)
        # FIXME Might need adjust, will come back to it later
        self.ee_link_offset = wp.vec3(0.0, 0.0, -0.1)
        # FIXME
        # Need to check the actual joint index of ant, 
        # could be very wrong
        self.ee_link_index = wp.array([2,4,6,8], dtype=wp.int32)
        # FIXME
        # 2 joints for each leg, could be wrong
        self.dof = 15

        self.target_origin = []
        for i in range(self.num_envs):
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(
                    wp.vec3(i * 5.0, 0.0, 0.0), wp.quat_identity()
                ),
            )

            foot_targets = [
                (i * 5.0 + 2, -2, -0),   # Target for back right foot (slightly forward on x-axis)
                (i * 5.0 - 2, 2, 0),  # Target for front left foot (slightly back on x-axis)
                (i * 5.0 + 2, 2, 0),  # Target for front right foot (slightly forward on x-axis)
                (i * 5.0 - 2, -2, 0)  # Target for back left foot (slightly back on x-axis)
            ]
            self.target_origin.append(foot_targets)
            # joint initial positions
            joint_q_value = np.array([0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]) + np.random.uniform(-0.2, 0.2, size=8)
            joint_q = np.clip(joint_q_value, -1.0, 1.0)
            builder.joint_q[-8:] = joint_q

        self.target_origin = np.array(self.target_origin)
        # finalize model
        self.model = builder.finalize()
        #self.model.ground = True

        self.model.joint_q.requires_grad = True
        self.model.body_q.requires_grad = True
        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0

        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

        self.ee_pos = wp.zeros(self.num_envs*4, dtype=wp.vec3, requires_grad=True)
        self.state_0 = self.model.state(requires_grad=True)
        self.state_1 = self.model.state(requires_grad=True)
        self.targets = self.target_origin.copy()
        self.profiler = {}

    def compute_ee_position(self):
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)
        wp.launch(
            compute_endeffector_position,
            dim=self.num_envs,
            inputs=[self.state_0.body_q, self.num_links, self.ee_link_index, self.ee_link_offset],
            outputs=[self.ee_pos],
        )
        return self.ee_pos
    
    # I hope this fucker works...
    def compute_jacobian(self):
        jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
        tape = wp.Tape()
        with tape:
            self.compute_ee_position()

        # Loop over the 3 output dimensions (x, y, z) for each end-effector
        for output_index in range(3):
            # Prepare a selection vector to compute the Jacobian for one dimension at a time
            select_index = np.zeros(3)
            select_index[output_index] = 1.0  # Select x, y, or z dimension
            # Repeat the selection vector for all environments
            e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.vec3)
            # Compute the gradient of the selected component of ee_pos with respect to joint_q
            tape.backward(grads={self.ee_pos: e})
            # Extract the gradients with respect to the joint positions
            q_grad_all = tape.gradients[self.model.joint_q]

            # Store the result in the corresponding row of the Jacobian
            jacobians[:, output_index, :] = q_grad_all.numpy().reshape(self.num_envs, self.dof)
            # Reset the tape for the next backward pass
            tape.zero()

        return jacobians
    
    def step(self):
        step_size = self.step_size
        # 1. Compute the Jacobian for each foot (end-effector)
        with wp.ScopedTimer("jacobian", print=False, active=True, dict=self.profiler):
            jacobians = self.compute_jacobian()
        
        # 2. Compute the current foot positions and the error with respect to target positions
        self.ee_pos_np = self.compute_ee_position().numpy()  # Current end-effector positions
        self.ee_pos_np = self.ee_pos_np.reshape(self.num_envs, 4, 3)

        error = self.targets - self.ee_pos_np  # Difference between target and current position
        self.error = error.reshape(self.num_envs, 4, 3, 1)  # Reshape: num_envs, 4 legs, 3 coords (x,y,z)

        delta_q = np.zeros((self.num_envs, self.dof), dtype=np.float32)
        # Map each leg's 2 DOF in the 8-DOF matrix
        for leg in range(4):  # Loop over each leg (4 legs per Ant)
            # Compute the update for joint angles using the Jacobian transpose
            delta_q += np.matmul(jacobians.transpose(0, 2, 1), 
                                    self.error[:, leg, :, :]).squeeze()
    

        # 4. Update joint positions
        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + self.step_size * delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=True,
        )
        
        self.state_0.clear_forces()
        wp.sim.collide(self.model, self.state_0)
        for _ in range(self.sim_substeps):
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            # Swap states after each substep to progress the simulation
            self.state_0, self.state_1 = self.state_1, self.state_0

        
        
    def render(self):
        if self.renderer is None:
            return

    # Update the foot target positions to move forward along the z-axis
    def update_target_positions(self, step_size):
        for i in range(self.num_envs):  # Loop over each Ant
            for j in range(4):  # Loop over each foot (4 feet)
                # Move the target position forward along the z-axis
                self.targets[i, j][1] += step_size  # Increase the z-value to move forward



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
    parser.add_argument("--train_iters", type=int, default=50, help="Total number of training iterations.")
    parser.add_argument("--num_envs", type=int, default=1, help="Total number of simulated environments.")
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=1,
        help="Total number of rollouts. In each rollout, a new set of target points is resampled for all environments.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        ant_agent = Ant(stage_path=args.stage_path, num_envs=args.num_envs)

        for episode in range(args.num_rollouts):
            print(f"Starting episode {episode + 1}/{args.num_rollouts}")
            # Reset the Ant agent at the beginning of each episode if needed
            ant_agent.targets = ant_agent.target_origin.copy()

            for iter in range(args.train_iters):
                # Update the target positions to simulate forward movement
                ant_agent.update_target_positions(0.2)
                ant_agent.step()
                ant_agent.render()
                print("iter:", iter, "error:", ant_agent.error.mean())

        if ant_agent.renderer:
            ant_agent.renderer.save()

        avg_time = np.array(ant_agent.profiler["jacobian"]).mean()
        avg_steps_second = 1000.0 * float(ant_agent.num_envs) / avg_time

        print(f"envs: {ant_agent.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")



