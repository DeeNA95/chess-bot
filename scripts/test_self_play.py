"""
Integration test for self-play training.
Runs a short training loop to verify everything works together.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_self_play_env():
    """Quick test of self-play environment."""
    from src.core.self_play_env import SelfPlayChessEnv

    print('Testing SelfPlayChessEnv...')
    env = SelfPlayChessEnv()
    obs, info = env.reset()

    print(f'  Observation shape: {obs["observation"].shape}')
    print(f'  Action mask shape: {obs["action_mask"].shape}')
    print(f'  Initial turn: {"White" if info["turn"] else "Black"}')

    # Play a few moves
    for i in range(10):
        mask = obs['action_mask']
        valid_actions = torch.where(mask)[0]
        if len(valid_actions) == 0:
            print(f'  Game over at move {i}')
            break
        action = valid_actions[torch.randint(len(valid_actions), (1,))].item()
        obs, reward, done, _, info = env.step(action)
        print(f'  Move {i+1}: reward={reward:.2f}, done={done}, turn={"White" if env.current_player else "Black"}')
        if done:
            break

    print('  SelfPlayChessEnv OK!')


def test_vector_self_play_env():
    """Quick test of vectorized self-play environment."""
    from src.core.vector_self_play_env import VectorSelfPlayEnv

    print('Testing VectorSelfPlayEnv...')
    env = VectorSelfPlayEnv(num_envs=4)
    obs, infos = env.reset()

    print(f'  Batched observation shape: {obs["observation"].shape}')
    print(f'  Batched action mask shape: {obs["action_mask"].shape}')

    # Take a few steps
    for step in range(5):
        masks = obs['action_mask']
        actions = []
        for i in range(4):
            valid = torch.where(masks[i])[0]
            action = valid[torch.randint(len(valid), (1,))].item()
            actions.append(action)

        actions_tensor = torch.tensor(actions)
        obs, rewards, dones, _, infos = env.step(actions_tensor)
        print(f'  Step {step+1}: rewards={rewards.tolist()}, dones={dones.tolist()}')

    env.close()
    print('  VectorSelfPlayEnv OK!')


def test_short_training():
    """Run a very short training loop to verify integration."""
    from src.core.vector_self_play_env import VectorSelfPlayEnv
    from src.agents.ppo_agent import PPOAgent
    from src.training.buffer import RolloutBuffer

    print('Testing short self-play training loop...')

    device = 'cpu'
    num_envs = 4
    rollout_steps = 8

    env = VectorSelfPlayEnv(num_envs=num_envs, device=device)
    agent = PPOAgent(device=device)
    buffer = RolloutBuffer(rollout_steps * num_envs, (116, 8, 8), (4096,), device=device)

    obs, _ = env.reset()

    # Collect rollout
    for _ in range(rollout_steps):
        with torch.no_grad():
            t_obs = obs['observation']
            t_mask = obs['action_mask']
            action, logprob, entropy, value = agent.get_action_and_value(t_obs, t_mask)

        next_obs, rewards, dones, _, infos = env.step(action)

        for i in range(num_envs):
            buffer.add(t_obs[i], action[i], logprob[i], rewards[i], dones[i], value[i], t_mask[i])

        obs = next_obs

    # Run one training step
    metrics = agent.train_step(buffer)

    print(f'  Rollout collected: {rollout_steps * num_envs} steps')
    print(f'  Training metrics: loss={metrics["loss"]:.4f}')

    env.close()
    print('  Short training loop OK!')


if __name__ == '__main__':
    print('=' * 50)
    print('Self-Play Integration Tests')
    print('=' * 50)

    try:
        test_self_play_env()
        print()
        test_vector_self_play_env()
        print()
        test_short_training()
        print()
        print('=' * 50)
        print('All integration tests passed!')
        print('=' * 50)
    except Exception as e:
        print(f'Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        exit(1)
