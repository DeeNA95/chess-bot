import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import train_loop
import torch

def test_short_run():
    print("Starting short verification run...")
    # Run for a tiny amount of steps to verify integration
    # total_timesteps = 100, which is < rollout_steps (2048),
    # but the loop expects to fill a rollout.
    # Let's override rollout_steps locally or just run enough steps.

    # We will run for 100 steps.
    # To see it work, we might need to reduce rollout_steps in trainer.py
    # or just run for 2100 steps.

    try:
        # We can use a small total_timesteps to just check if it launches
        train_loop(total_timesteps=2100, checkpoint_dir="./test_checkpoints")
        print("Short run completed successfully!")
    except Exception as e:
        print(f"Short run failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_short_run()
