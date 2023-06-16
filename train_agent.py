import numpy

from src.config import AgentConfig, EnvironmentConfig
from src.piece import Piece
from src.environment import KanoodleEnvironment


def train_agent(agent_config: AgentConfig, environment_config: EnvironmentConfig):
    print(agent_config)
    print(environment_config)

    environment = KanoodleEnvironment(environment_config)

    action_confs = numpy.array(range(len(environment.actions)), dtype=numpy.float32)

    is_finished = False
    while not is_finished:
        observation, reward, is_finished, info = environment.step(action_confs)
        environment.render(mode="console")


if __name__ == "__main__":
    agent_config = AgentConfig()
    environment_config = EnvironmentConfig(pieces_set_name="test")

    train_agent(agent_config, environment_config)
