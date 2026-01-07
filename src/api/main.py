from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chess
import torch
import os
from src.agents.ppo_agent import PPOAgent
from src.core.state_encoder import StateEncoder

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Agent and Encoder
agent = None
encoder = None
device = "cpu"

class MoveRequest(BaseModel):
    fen: str

class MoveResponse(BaseModel):
    uci: str
    san: str
    evaluation: float = 0.0

@app.on_event("startup")
async def startup_event():
    global agent, device, encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")

    agent = PPOAgent(device=device)
    encoder = StateEncoder(device=device)

    # Load checkpoint
    checkpoint_path = "checkpoints/mcts_final.pt"
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("Warning: No checkpoint found! specific path: checkpoints/ppo_final.pt")

@app.post("/move", response_model=MoveResponse)
async def predict_move(req: MoveRequest):
    board = chess.Board(req.fen)

    obs = encoder.encode(board).unsqueeze(0)  # Batch dim

    # Get action mask - match training logic exactly
    mask = torch.zeros(1, 4096, dtype=torch.bool, device=device)

    # Filter moves the same way as training: only Queen promotions allowed
    valid_indices = [
        move.from_square * 64 + move.to_square
        for move in board.legal_moves
        if not move.promotion or move.promotion == chess.QUEEN
    ]
    if valid_indices:
        mask[0, valid_indices] = True

    with torch.no_grad():
        # Get raw logits to debug
        logits, value = agent.model(obs)
        logits[~mask] = -float('inf')
        probs = torch.softmax(logits, dim=-1)

        # Debug: show top 5 moves by probability
        top_probs, top_indices = torch.topk(probs[0], min(5, len(valid_indices)))
        print(f"\n=== Move Request: {req.fen[:30]}... ===")
        print(f"Legal moves: {len(valid_indices)}")
        print(f"Top 5 moves by probability:")
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            from_sq = idx // 64
            to_sq = idx % 64
            m = chess.Move(from_sq, to_sq)
            try:
                san = board.san(m)
            except:
                san = m.uci()
            print(f"  {san}: {prob:.4f} ({prob*100:.1f}%)")

        # Check if probabilities are uniform (untrained model)
        valid_probs = probs[0, valid_indices]
        prob_std = valid_probs.std().item()
        prob_max = valid_probs.max().item()
        print(f"Prob std: {prob_std:.6f}, max: {prob_max:.4f}")
        if prob_std < 0.01:
            print("⚠️ WARNING: Probabilities are nearly uniform - model may not have learned!")

        # Use deterministic=True for best move selection during gameplay
        action_tensor, _, _, value = agent.get_action_and_value(obs, mask, deterministic=True)
        action = action_tensor.item()

    move = chess.Move(int(action // 64), int(action % 64))

    # Handle pawn promotion (always Queen to match training)
    if chess.square_rank(move.to_square) in [0, 7]:
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            move.promotion = chess.QUEEN

    print(f"Selected: {board.san(move)}")

    return MoveResponse(
        uci=move.uci(),
        san=board.san(move),
        evaluation=value.item()
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

    # Check if weights are initialized (not random)
    first_layer = list(agent.model.parameters())[0]
    weight_stats = {
        "mean": first_layer.mean().item(),
        "std": first_layer.std().item(),
        "min": first_layer.min().item(),
        "max": first_layer.max().item(),
    }

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "first_layer_stats": weight_stats,
        "device": str(device),
    }
