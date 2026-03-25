import numpy as np

class SwarmOptimizer:
    def __init__(self, population_size, num_dimensions, fitness_function):
        self.population_size = population_size
        self.num_dimensions = num_dimensions
        self.fitness_function = fitness_function
        self.population = self.initialize_population()
        self.personal_best = self.population.copy()
        self.global_best = self.population[0].copy()
        self.velocities = np.zeros((population_size, num_dimensions))
        self.cognitive_weight = 2.0
        self.social_weight = 2.0
        self.inertia_weight = 0.5

    def initialize_population(self):
        population = np.random.uniform(-10, 10, (self.population_size, self.num_dimensions))
        return population

    def update_velocities(self):
        r1 = np.random.uniform(0, 1, (self.population_size, self.num_dimensions))
        r2 = np.random.uniform(0, 1, (self.population_size, self.num_dimensions))
        self.velocities = self.inertia_weight * self.velocities + \\
                         self.cognitive_weight * r1 * (self.personal_best - self.population) + \\
                         self.social_weight * r2 * (self.global_best - self.population)

    def update_positions(self):
        self.population += self.velocities

    def update_personal_best(self):
        for i in range(self.population_size):
            if self.fitness_function(self.population[i]) < self.fitness_function(self.personal_best[i]):
                self.personal_best[i] = self.population[i].copy()

    def update_global_best(self):
        best_index = np.argmin([self.fitness_function(x) for x in self.population])
        self.global_best = self.population[best_index].copy()

    def optimize(self, num_iterations):
        for _ in range(num_iterations):
            self.update_velocities()
            self.update_positions()
            self.update_personal_best()
            self.update_global_best()
        return self.global_best
