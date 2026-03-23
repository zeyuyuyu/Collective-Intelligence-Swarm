import numpy as np
from typing import Callable, Tuple

class ParticleSwarmOptimizer:
    def __init__(
        self,
        n_particles: int = 30,
        dimensions: int = 2,
        c1: float = 2.0,
        c2: float = 2.0,
        w: float = 0.7,
        bounds: Tuple[float, float] = (-1.0, 1.0)
    ):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient
        self.w = w    # Inertia weight
        self.bounds = bounds
        
        # Initialize particle positions and velocities
        self.positions = np.random.uniform(
            bounds[0], bounds[1], 
            (n_particles, dimensions)
        )
        self.velocities = np.random.uniform(
            -1, 1, 
            (n_particles, dimensions)
        )
        
        # Initialize personal best positions and global best
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(n_particles, np.inf)
        self.gbest_position = np.zeros(dimensions)
        self.gbest_score = np.inf

    def optimize(
        self,
        objective_func: Callable[[np.ndarray], float],
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ) -> Tuple[np.ndarray, float]:
        """Execute particle swarm optimization algorithm.
        
        Args:
            objective_func: Function to minimize
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (best position found, best score found)
        """
        prev_best = np.inf
        
        for _ in range(max_iterations):
            # Evaluate current positions
            scores = np.array([objective_func(p) for p in self.positions])
            
            # Update personal bests
            improved = scores < self.pbest_scores
            self.pbest_scores[improved] = scores[improved]
            self.pbest_positions[improved] = self.positions[improved]
            
            # Update global best
            min_score_idx = np.argmin(scores)
            if scores[min_score_idx] < self.gbest_score:
                self.gbest_score = scores[min_score_idx]
                self.gbest_position = self.positions[min_score_idx].copy()
            
            # Check convergence
            if abs(prev_best - self.gbest_score) < tolerance:
                break
            prev_best = self.gbest_score
            
            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            self.velocities = (
                self.w * self.velocities +
                self.c1 * r1 * (self.pbest_positions - self.positions) +
                self.c2 * r2 * (self.gbest_position - self.positions)
            )
            
            self.positions += self.velocities
            
            # Enforce bounds
            self.positions = np.clip(
                self.positions, 
                self.bounds[0], 
                self.bounds[1]
            )
            
        return self.gbest_position, self.gbest_score

    def reset(self):
        """Reset the optimizer state."""
        self.__init__(
            self.n_particles,
            self.dimensions,
            self.c1,
            self.c2,
            self.w,
            self.bounds
        )