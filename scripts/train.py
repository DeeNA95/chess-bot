
import sys
import os
import signal
import argparse

# Add src to pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.training.trainer import train_loop

def main():
    parser = argparse.ArgumentParser(description="Train the Chess Bot")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nTraining interrupted. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(f"Starting training with config: {args.config}")
    train_loop(args.config)

if __name__ == "__main__":
    main()
