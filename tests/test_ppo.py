import os
import sys
import torch
import chess
# Fix sys.path to root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rl.ppo import PPO
from src.core.config import AppConfig
from src.agents.ppo_agent import ChessAgent
from src.core.state_encoder import StateEncoder

def test_ppo_step():
    device = "cpu"
    config = AppConfig.load()
    config.algorithm = "ppo"

    agent = ChessAgent(device=device)
    encoder = StateEncoder(device=device)
    ppo = PPO(config, agent.model)

    # Create a dummy batch
    batch_size = 4
    obs_list = []
    actions = []
    old_log_probs = []
    rewards = []
    values = []
    masks = []

    board = chess.Board()
    obs = encoder.encode(board)
    mask = encoder.get_action_mask(board)

    for i in range(batch_size):
        obs_list.append(obs)
        actions.append(0)
        old_log_probs.append(-1.0)
        rewards.append(1.0) # Dummy MC return
        values.append(0.5)  # Dummy value
        masks.append(mask)

    obs_batch = torch.stack(obs_list)
    actions_batch = torch.tensor(actions)
    old_log_probs_batch = torch.tensor(old_log_probs)
    rewards_batch = torch.tensor(rewards)
    advantages_batch = rewards_batch - torch.tensor(values) # Simple advantage
    masks_batch = torch.stack(masks)

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=1e-4)

    # Test compute_loss
    metrics = ppo.compute_loss(
        obs_batch,
        actions_batch,
        old_log_probs_batch,
        rewards_batch,
        advantages_batch,
        masks_batch
    )

    loss = metrics['loss']
    print(f"Loss: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() is not None
    print("PPO step test passed!")

if __name__ == "__main__":
    test_ppo_step()
