from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chess
import torch
import os
import numpy as np
from src.agents.ppo_agent import ChessAgent
from src.core.state_encoder import StateEncoder
from src.search.mcts import MCTS

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Agent, Encoder, and MCTS
agent = None
encoder = None
mcts = None
device = "cpu"

class MoveRequest(BaseModel):
    fen: str
    num_simulations: int = 50

class MoveResponse(BaseModel):
    uci: str
    san: str
    evaluation: float = 0.0
    win_probability: float = 0.5

@app.on_event("startup")
async def startup_event():
    global agent, device, encoder, mcts
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    agent = ChessAgent(device=device)
    encoder = StateEncoder(device=device)

    # Initialize MCTS with the loaded model
    # Note: num_simulations can be overridden per request if we instantiate per request,
    # or we just use a default here. For now, we'll instantiate MCTS here as a base
    # but the search method uses the class params.
    # Actually MCTS class stores num_simulations.
    # To allow per-request param, we might need to instantiate MCTS per request
    # or modify MCTS to accept `sims` in `search_batch`.
    # For efficiency, we will instantiate a global MCTS with a default,
    # and re-instantiate if needed or just accept the request param in the handler.
    mcts = MCTS(model=agent.model, encoder=encoder, device=device, num_simulations=50)

    # Load checkpoint
    checkpoint_path = "checkpoints/mcts_game_15712.pt"
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: No checkpoint found! specific path: {checkpoint_path}")

@app.post("/move", response_model=MoveResponse)
async def predict_move(req: MoveRequest):
    if not agent or not mcts:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        board = chess.Board(req.fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid FEN string")

    if board.is_game_over():
         raise HTTPException(status_code=400, detail="Game is already over")

    # Update simulations if requested
    current_mcts = mcts
    if req.num_simulations != mcts.num_simulations:
        # Create a temporary MCTS instance for this request if params differ
        current_mcts = MCTS(
            model=agent.model,
            encoder=encoder,
            device=device,
            num_simulations=req.num_simulations
        )

    # Run MCTS Search
    # search_batch expects a list of boards
    results = current_mcts.search_batch([board])
    policy, value = results[0]  # (tensor, float)

    # Select best move from policy (robust child / max visits)
    # policy is a tensor of shape (4096,) containing probabilities
    best_action_idx = torch.argmax(policy).item()

    # Decode action to move
    from_sq = int(best_action_idx // 64)
    to_sq = int(best_action_idx % 64)
    move = chess.Move(from_sq, to_sq)

    # Handle promotions (AlphaZero simplified encoding usually implies Queen)
    # Check if this move is a promotion candidate
    if chess.square_rank(to_sq) in [0, 7]:
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            move.promotion = chess.QUEEN

    # Verify legality (just in case)
    if move not in board.legal_moves:
        # Fallback: if MCTS picked an illegal move (rare but possible if masked incorrectly or bug),
        # pick the legal move with highest policy value.
        legal_moves = list(board.legal_moves)
        best_legal_move = None
        best_prob = -1.0

        for m in legal_moves:
            # Map move to action idx
            idx = m.from_square * 64 + m.to_square
            prob = policy[idx].item()
            if prob > best_prob:
                best_prob = prob
                best_legal_move = m

        if best_legal_move:
            move = best_legal_move
            print(f"⚠️ MCTS selected illegal move {chess.Move(from_sq, to_sq)}, fell back to {move}")
        else:
             # Should never happen
             move = legal_moves[0]

    # Calculate win probability from value (-1 to 1) -> (0 to 1)
    win_prob = (value + 1) / 2

    return MoveResponse(
        uci=move.uci(),
        san=board.san(move),
        evaluation=float(value),
        win_probability=float(win_prob)
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/model")
def debug_model():
    """Debug endpoint to check model state."""
    if agent is None:
        return {"error": "Agent not loaded"}

    # Count parameters
    total_params = sum(p.numel() for p in agent.model.parameters())
    trainable_params = sum(p.numel() for p in agent.model.parameters() if p.requires_grad)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "device": str(device),
        "checkpoint": "mcts_game_112.pt"
    }
