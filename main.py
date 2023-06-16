from src.config import AgentConfig, EnvironmentConfig
from src.train_model import train_model

# TODO: argparse


if __name__ == "__main__":
    agent_config = AgentConfig()
    environment_config = EnvironmentConfig()

    train_model(agent_config, environment_config)
