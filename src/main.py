import os
import sys
import time
import random

from cis.agent import Agent
from cis.swarm import Swarm
from cis.environment import Environment
from cis.communication import CommunicationManager

def main():
    # Initialize the environment
    env = Environment()

    # Create the communication manager
    comm_manager = CommunicationManager()

    # Spawn the agent swarm
    swarm = Swarm(num_agents=100, env=env, comm_manager=comm_manager)

    # Run the swarm simulation
    while True:
        swarm.update()
        time.sleep(0.1)

if __name__ == '__main__':
    main()