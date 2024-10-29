import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import copy


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

        self.num_envs = num_envs
        offsets = compute_env_offsets(self.num_envs)
        self.dof = len(articulation_builder.joint_q)

        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

            builder.joint_q[-8:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            builder.joint_axis_mode = [wp.sim.JOINT_MODE_TARGET_POSITION] * len(builder.joint_axis_mode)
            builder.joint_act[-8:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]


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
        

        self.num_links = len(articulation_builder.joint_type)
        # FIXME
        # Might need to change to [2,4,6,8] for 4 feets
        self.ee_link_index = 4
        # FIXME
        # Currently no offset, might need to change
        self.ee_link_offset = wp.vec3(0.0, 0.0, 0.0)
        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True)
        
        #tape_1 = wp.Tape()
        #with tape_1:
        #    wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        # FIXME
        # Might not want to call it here
        # It seems like the end effector position is a relative position wrt its body
        # Try add 0.01 to the x axis of ee_pos for now
        '''
        tape = wp.Tape()
        with tape:
            self.compute_ee_position()
        self.target_origin = self.ee_pos
        self.target_origin = np.array(self.target_origin)
        self.targets = self.target_origin.copy()
        '''

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

    def compute_jacobian(self):

        # jacobians -> [num_envs][3][15]

        #print(jacobians)
        return jacobians



    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()


            
            # self.state_0.joint_q = wp.array(
            #     self.state_0.joint_q.numpy() + [0,0,0,0,0,0,0,0.01,0,0,0,0,0,0,0,
            #                                     0,0,0,0,0,0,0,0.01,0,0,0,0,0,0,0],
            #     dtype=wp.float32,
            #     requires_grad=True,
            # ) 
            

            #self.compute_jacobian()
            #print(self.ee_pos)
            tape = wp.Tape()
            with tape:
                wp.sim.collide(self.model, self.state_0)
                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.compute_ee_position()
                print(self.ee_pos)
                jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
                for output_index in range(3):
                    # select which row of the Jacobian we want to compute
                    select_index = np.zeros(3)
                    select_index[output_index] = 1.0
                    e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.vec3)
                    #print(e)
                    tape.backward(grads={self.ee_pos: e})
                    if self.state_0.joint_q in tape.gradients:
                        q_grad_i = tape.gradients[self.state_0.joint_q]
                    else:
                        print("joint_q gradients not found in tape.")
                    jacobians[:, output_index, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)
                    tape.zero()
                #print(jacobians)
            self.state_0, self.state_1 = self.state_1, self.state_0

    
    def step(self):
        with wp.ScopedTimer("step", print=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
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

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()