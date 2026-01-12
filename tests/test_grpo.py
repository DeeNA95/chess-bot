import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import chess
from src.rl.grpo import GRPO
from src.core.config import AppConfig
from src.agents.ppo_agent import ChessAgent
from src.core.state_encoder import StateEncoder

def test_grpo_step():
    device = "cpu"
    config = AppConfig.load()
    config.algorithm = "grpo"

    agent = ChessAgent(device=device)
    encoder = StateEncoder(device=device)
    grpo = GRPO(config, agent.model)

    # Create a dummy batch of 16 (1 group of 16)
    group_size = config.grpo.group_size
    obs_list = []
    actions = []
    old_log_probs = []
    rewards = []
    masks = []

    board = chess.Board()
    obs = encoder.encode(board)
    mask = encoder.get_action_mask(board)

    for i in range(group_size):
        obs_list.append(obs)
        actions.append(0) # dummy action
        old_log_probs.append(-1.0) # dummy log prob
        rewards.append(float(i % 2)) # alternating rewards to test variance
        masks.append(mask)

    obs_batch = torch.stack(obs_list)
    actions_batch = torch.tensor(actions)
    old_log_probs_batch = torch.tensor(old_log_probs)
    rewards_batch = torch.tensor(rewards)
    masks_batch = torch.stack(masks)

    # optimizer
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=1e-4)

    # forward
    metrics = grpo.compute_loss(
        obs_batch,
        actions_batch,
        old_log_probs_batch,
        rewards_batch,
        masks_batch
    )

    loss = metrics['loss']
    print(f"Loss: {loss.item()}")

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() is not None
    print("GRPO step test passed!")

if __name__ == "__main__":
    test_grpo_step()
