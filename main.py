from src.config import Config
from src.train_model import train_model

# TODO: argparse


if __name__ == "__main__":
    config = Config()

    train_model(config)
