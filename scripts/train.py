import argparse

from src.training.trainer import train_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chess-bot training.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file.",
    )
    args = parser.parse_args()

    train_loop(args.config)


if __name__ == "__main__":
    main()
