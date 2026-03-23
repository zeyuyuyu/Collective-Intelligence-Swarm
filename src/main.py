import numpy as np
from typing import List, Tuple

class SwarmAgent:
    def __init__(self, position: Tuple[float, float], velocity: Tuple[float, float]):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.neighbors = []
        self.target_position = None

    def update_position(self, dt: float):
        self.position += self.velocity * dt

    def update_velocity(self, neighbors: List['SwarmAgent'], target_position: Tuple[float, float], cohesion_weight: float, alignment_weight: float, separation_weight: float):
        self.neighbors = neighbors
        self.target_position = np.array(target_position)

        # Cohesion
        cohesion_force = np.zeros(2)
        for neighbor in self.neighbors:
            cohesion_force += neighbor.position
        cohesion_force /= len(self.neighbors)
        cohesion_force -= self.position
        cohesion_force *= cohesion_weight

        # Alignment
        alignment_force = np.zeros(2)
        for neighbor in self.neighbors:
            alignment_force += neighbor.velocity
        alignment_force /= len(self.neighbors)
        alignment_force *= alignment_weight

        # Separation
        separation_force = np.zeros(2)
        for neighbor in self.neighbors:
            diff = self.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                separation_force += diff / dist
        separation_force *= separation_weight

        # Target attraction
        target_force = self.target_position - self.position

        self.velocity += cohesion_force + alignment_force + separation_force + target_force

class SwarmSimulation:
    def __init__(self, num_agents: int, world_size: Tuple[float, float]):
        self.agents = [SwarmAgent((np.random.uniform(0, world_size[0]), np.random.uniform(0, world_size[1])), (np.random.uniform(-1, 1), np.random.uniform(-1, 1))) for _ in range(num_agents)]
        self.world_size = world_size
        self.cohesion_weight = 0.1
        self.alignment_weight = 0.1
        self.separation_weight = 0.5
        self.target_position = (world_size[0] / 2, world_size[1] / 2)

    def step(self, dt: float):
        for agent in self.agents:
            neighbors = [a for a in self.agents if np.linalg.norm(a.position - agent.position) < 50]
            agent.update_velocity(neighbors, self.target_position, self.cohesion_weight, self.alignment_weight, self.separation_weight)
            agent.update_position(dt)

            # Wrap around world boundaries
            agent.position[0] = (agent.position[0] + self.world_size[0]) % self.world_size[0]
            agent.position[1] = (agent.position[1] + self.world_size[1]) % self.world_size[1]

if __name__ == '__main__':
    simulation = SwarmSimulation(100, (1000, 1000))
    dt = 0.1
    while True:
        simulation.step(dt)