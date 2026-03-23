import random
import networkx as nx
import numpy as np

class SwarmIntelligence:
    def __init__(self, num_agents, communication_range):
        self.num_agents = num_agents
        self.communication_range = communication_range
        self.agents = [Agent(i) for i in range(num_agents)]
        self.graph = self.build_communication_graph()

    def build_communication_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.num_agents))
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if np.linalg.norm(self.agents[i].position - self.agents[j].position) <= self.communication_range:
                    G.add_edge(i, j)
        return G

    def run_consensus(self, max_iterations=100):
        for _ in range(max_iterations):
            for agent in self.agents:
                agent.update_state(self.graph)
        return [agent.state for agent in self.agents]

class Agent:
    def __init__(self, id):
        self.id = id
        self.state = random.uniform(0, 1)
        self.position = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])

    def update_state(self, graph):
        neighbors = list(graph.neighbors(self.id))
        if neighbors:
            self.state = np.mean([self.agents[n].state for n in neighbors])
