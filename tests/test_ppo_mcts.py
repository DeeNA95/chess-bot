import pytest
import torch
import chess
from src.core.config import AppConfig, RewardSettings
from src.core.state_encoder import StateEncoder
from src.agents.ppo_agent import ChessAgent
from src.search.mcts import MCTS
from src.training.verifiers import ChessRubric
from src.training.async_verifier import AsyncStockfishVerifier
from src.training.trainer import play_games_ppo_mcts, PPOSample

@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.training.num_parallel_games = 1
    cfg.mcts.num_simulations = 2
    cfg.rewards.stockfish_depth = 1 # Fast for test
    return cfg

@pytest.fixture
def setup_components(config):
    device = torch.device('cpu')
    agent = ChessAgent(device=str(device))
    encoder = StateEncoder(device=str(device))
    mcts = MCTS(agent.model, encoder, str(device), num_simulations=config.mcts.num_simulations)

    rubric = ChessRubric()
    # Mock stockfish verifier to avoid binary dependency in unit test?
    # Or just try to use it if available.
    try:
        rubric.add_verifier(
            AsyncStockfishVerifier("/opt/homebrew/bin/stockfish", depth=1, num_workers=1),
            weight=1.0
        )
    except:
        pass # If stockfish fails, we still test structure

    return agent, mcts, rubric, encoder, device

def test_play_games_ppo_mcts(config, setup_components):
    agent, mcts, rubric, encoder, device = setup_components

    # Run only 2 moves
    samples = play_games_ppo_mcts(
        agent, mcts, rubric, encoder, config, device, max_moves=2
    )

    assert len(samples) > 0
    assert isinstance(samples[0], PPOSample)
    assert samples[0].log_prob < 0 # Log prob should be negative
    # Check if reward is populated (rubric might return 0 if no stockfish, but structure holds)
    print(f"Sample reward: {samples[0].reward}")
