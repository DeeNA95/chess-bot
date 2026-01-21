import pytest
import torch
import chess
from src.core.config import AppConfig
from src.core.state_encoder import StateEncoder
from src.agents.ppo_agent import ChessAgent
from src.search.mcts import MCTS
from src.training.verifiers import ChessRubric
from src.training.async_verifier import AsyncStockfishVerifier
from src.training.trainer import play_games_grpo_mcts, GRPOSample

@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.training.num_parallel_games = 1
    cfg.mcts.num_simulations = 2
    cfg.grpo.group_size = 2 # Small group
    cfg.rewards.stockfish_depth = 1
    return cfg

@pytest.fixture
def setup_components(config):
    device = torch.device('cpu')
    agent = ChessAgent(device=str(device))
    encoder = StateEncoder(device=str(device))
    mcts = MCTS(agent.model, encoder, str(device), num_simulations=config.mcts.num_simulations)

    rubric = ChessRubric()
    try:
        rubric.add_verifier(
            AsyncStockfishVerifier("/opt/homebrew/bin/stockfish", depth=1, num_workers=1),
            weight=1.0
        )
    except:
        pass

    return agent, mcts, rubric, encoder, device

def test_play_games_grpo_mcts(config, setup_components):
    agent, mcts, rubric, encoder, device = setup_components

    # Run loop
    samples = play_games_grpo_mcts(
        agent, mcts, rubric, encoder, config, device, max_moves=2
    )

    assert len(samples) > 0
    # Should be multiple of group_size
    assert len(samples) % config.grpo.group_size == 0
    assert isinstance(samples[0], GRPOSample)
    print(f"GRPO Sample Reward: {samples[0].reward}")
