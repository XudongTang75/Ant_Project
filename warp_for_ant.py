import numpy as np
import torch
import torch.optim as optim
import warp as wp
from Model.fnn import PolicyNetwork
from Ant.ant import Example


def train_policy(num_episodes, num_steps, learning_rate, path):

    # Create the environment instance
    env = Example(stage_path="ant.usd", num_envs=1)  # Use a single environment for training
    state_dim = len(env.state_0.joint_q) + len(env.state_0.joint_qd)  # Calculate the correct state dimension
    action_dim = len(env.model.joint_act)  # Calculate the correct action dimension

    # Initialize the policy network and optimizer
    policy = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    best_reward = float('-inf')  # Initialize with negative infinity to track the best reward

    for episode in range(num_episodes):
        optimizer.zero_grad()

        # Reset environment to the initial state
        env.state_0 = env.model.state()
        env.state_1 = env.model.state()
        wp.sim.eval_fk(env.model, env.model.joint_q, env.model.joint_qd, None, env.state_0)

        total_reward = torch.tensor(0.0, requires_grad=True)  # Initialize total_reward as a PyTorch tensor

        # Run the simulation for a specified number of steps
        for step in range(num_steps):
            # Get the current state as a tensor for the policy network
            # Assume state is represented by joint positions and velocities concatenated
            state_tensor = torch.tensor(
                np.concatenate((env.state_0.joint_q, env.state_0.joint_qd)), 
                dtype=torch.float32,
                requires_grad=True
            )

            # Generate actions using the policy network
            actions = policy(state_tensor)
            # Simulate the environment with the generated actions
            env.step(actions.detach().numpy())

            # Compute reward and accumulate it
            reward = torch.tensor(env.compute_reward(), dtype=torch.float32, requires_grad=True)
            total_reward = total_reward + reward

        # Define loss as the negative total reward (to maximize reward)
        loss = -total_reward
        # Backpropagate the loss
        loss.backward()
        # Update policy network parameters
        optimizer.step()

        # Check if this is the best reward so far
        if total_reward > best_reward:
            best_reward = total_reward
            # Save the current model as the best one
            print(f"New best reward: {best_reward} at episode {episode + 1}")
            if env.renderer:
                env.renderer.save(stage_path=path)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


# Main
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--num_steps", type=int, default=300, help="Number of steps per episode.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--best_stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="best_ant.usd",
        help="Path to save the best USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=1, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    train_policy(
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        path=args.best_stage_path,
    )

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.best_stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
