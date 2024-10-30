import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render


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




@wp.kernel
def compute_endeffector_position(
    body_q: wp.array(dtype=wp.transform),   # The transformation of the entire body
    num_links: int,                         # Number of links (joints) in each leg
    # FIXME
    # Only use the front left as end effectors for now, might need to change to 4
    ee_link_indices: int,   # INdex of the last link in the chain for each leg
    ee_link_offsets: wp.vec3,  # Offsets for each end-effector relative to its leg's last joint
    ee_pos: wp.array(dtype=wp.vec3)         # Output array for end-effector positions (all 4 feet per Ant)
):
    tid = wp.tid()  # Thread ID for each Ant
    # Compute positions for all 4 end-effectors (feet) for this Ant
    ee_pos[tid] = wp.transform_point(
        body_q[tid * num_links + ee_link_indices],  # Get the transformation of the last link in each leg
        ee_link_offsets  # Apply the offset for each foot
    )


class Example:
    def __init__(self, stage_path="ant.usd", num_envs=1):
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
        # duration of each simulation frame in seconds
        self.frame_dt = 1.0 / fps

        # number of substeps per frame
        self.sim_substeps = 5
        # timestep for each substep
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.step_size = 0.2

        self.num_envs = num_envs
        self.offsets = compute_env_offsets(self.num_envs)
        self.dof = len(articulation_builder.joint_q)

        self.target_origin = []
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(self.offsets[i], wp.quat_identity()))

            builder.joint_q[-8:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            builder.joint_axis_mode = [wp.sim.JOINT_MODE_FORCE] * len(builder.joint_axis_mode)
            builder.joint_act[-8:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]



        np.set_printoptions(suppress=True)
        

        self.model = builder.finalize()
        self.model.ground = True
        self.model.joint_q.requires_grad = True
        self.model.body_q.requires_grad = True
        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0


        self.integrator = wp.sim.FeatherstoneIntegrator(self.model)

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

        self.state_0 = self.model.state(requires_grad=True)
        self.state_1 = self.model.state(requires_grad=True)
        
        self.control = self.model.control()

        self.num_links = len(articulation_builder.joint_type)
        # FIXME
        # Might need to change to [2,4,6,8] for 4 feets
        self.ee_link_index = 8
        # FIXME
        # Currently no offset, might need to change
        self.ee_link_offset = wp.vec3(0.0, 0.0, 0.0)
        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True)
        
        self.targets = self.target_origin.copy()

        self.profiler = {}


        self.use_cuda_graph = wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None
        

    def compute_ee_position(self):
        wp.launch(
            compute_endeffector_position,
            dim=self.num_envs,
            inputs=[self.state_0.body_q, self.num_links, self.ee_link_index, self.ee_link_offset],
            outputs=[self.ee_pos],
        )

    # if multiple envs, change z value 
    # ee start position for joint #0: [0, 0.3, 0]
    # ee start position for joint #1: [0.2, 0.3, -0.2]
    # ee start position for joint #2: [0.4, 0.3, -0.4]
    # ee start position for joint #3: [-0.2, 0.3, -0.2]
    # ee start position for joint #4: [-0.4, 0.3, -0.4]
    # ee start position for joint #5: [-0.2, 0.3, 0.2]
    # ee start position for joint #6: [-0.4, 0.3, 0.4]
    # ee start position for joint #7: [0.2, 0.3, 0.2]
    # ee start position for joint #8: [0.4, 0.3, 0.4]

    

    def simulate(self, frame_num):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            #print(self.ee_pos)
            #print(self.target_origin)
            tape = wp.Tape()
            with tape:
                wp.sim.collide(self.model, self.state_0)
                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt, control=self.control)
                self.compute_ee_position()
                #print(self.state_0.joint_q)
                jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
                for output_index in range(3):
                    # select which row of the Jacobian we want to compute
                    select_index = np.zeros(3)
                    select_index[output_index] = 1.0
                    e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.vec3)
                    tape.backward(grads={self.ee_pos: e})
                    q_grad_i = tape.gradients[self.state_0.joint_q]
                    jacobians[:, output_index, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)
                    tape.zero()
        
            self.state_0, self.state_1 = self.state_1, self.state_0

            
            if frame_num > 60:
                
                ##################################################################################################
                #             Generate RANDOM action input, eventually want to do controlled action              #
                ##################################################################################################
                self.control.joint_act = wp.array(np.random.uniform(-500, 500, size=16).astype(np.float32))



                ##################################################################################################
                #                This part update joint_q using jacobian wrt the ee_position                     #
                # It does NOT have any action involved, so the code simply render the joint at updated location! #
                ##################################################################################################
                '''
                error = np.tile(np.array([0.008, 0, 0]), (self.num_envs, 1))

                self.error = error.reshape(self.num_envs, 3, 1)
                # compute Jacobian transpose update
                delta_q = np.matmul(jacobians.transpose(0, 2, 1), self.error)
                self.state_0.joint_q = wp.array(
                    self.state_0.joint_q.numpy() + self.step_size * delta_q.flatten(),
                    dtype=wp.float32,
                    requires_grad=True,
                )
                '''
                

    
    def step(self, frame_num):
        with wp.ScopedTimer("step", print=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate(frame_num)
        self.sim_time += self.frame_dt
        

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", print=False):
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
    parser.add_argument("--num_envs", type=int, default=2, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)
        

        for i in range(args.num_frames):
            #if i > 60:
            #    example.targets = example.ee_pos.numpy().copy()
            #    example.targets[:,0] += 0.0002
            example.step(i)
            example.render()

        if example.renderer:
            example.renderer.save()