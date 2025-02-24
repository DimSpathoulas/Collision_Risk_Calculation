from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.integrate import quad
from scipy.special import erf


@dataclass
class OctagonEdge:
    xl: float
    xu: float
    m: float
    b: float
    contribution: int = 1  # 1 for positive, -1 for negative


def split_octagon_segments_2(octagon_vertices: np.ndarray) -> List[OctagonEdge]:
    # Get centroid
    centroid = np.mean(octagon_vertices, axis=0)

    # Sort vertices by angle around centroid
    angles = np.arctan2(octagon_vertices[:, 1] - centroid[1],
                        octagon_vertices[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_vertices = octagon_vertices[sorted_idx]

    edges = []
    n = len(sorted_vertices)

    # Create edges while preserving octagon shape
    for i in range(n):
        x1, y1 = sorted_vertices[i]
        x2, y2 = sorted_vertices[(i + 1) % n]

        # Ignore vertical edges
        if abs(x2 - x1) < 1e-10:
            continue

        # Compute slope and intercept
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        # Determine contribution using cross-product approach
        vector1 = [x2 - x1, y2 - y1]
        vector2 = [centroid[0] - x1, centroid[1] - y1]
        cross_product = np.cross(vector1, vector2)
        contribution = 1 if cross_product > 0 else -1

        edges.append(OctagonEdge(
            xl=min(x1, x2),
            xu=max(x1, x2),
            m=m,
            b=b,
            contribution=contribution
        ))

    return edges


def split_octagon_segments(octagon_vertices: np.ndarray) -> List[OctagonEdge]:
    # Get centroid
    centroid = np.mean(octagon_vertices, axis=0)

    # Sort vertices by angle around centroid
    angles = np.arctan2(octagon_vertices[:, 1] - centroid[1],
                        octagon_vertices[:, 0] - centroid[0])
    sorted_idx = np.argsort(angles)
    sorted_vertices = octagon_vertices[sorted_idx]

    edges = []
    n = len(sorted_vertices)

    # Create edges while preserving octagon shape
    for i in range(n):
        x1, y1 = sorted_vertices[i]
        x2, y2 = sorted_vertices[(i + 1) % n]

        if abs(x2 - x1) < 1e-10:  # Vertical edge
            continue
            # m = 1e6 if y2 > y1 else -1e6
            # b = y1 - m * x1
        else:
            m = (y2 - y1) / (x2 - x1)
            b = y1 - m * x1

        # Determine contribution based on midpoint position
        mid_y = (y1 + y2) / 2
        contribution = 1 if mid_y > centroid[1] else -1

        edges.append(OctagonEdge(
            xl=min(x1, x2),
            xu=max(x1, x2),
            m=m,
            b=b,
            contribution=contribution
        ))

    return edges


class CSPCalculator:
    def __init__(self, sigma_x: float, sigma_y: float):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sqrt2 = np.sqrt(2)

    def _integrand(self, x: float, m: float, b: float) -> float:
        """Compute integrand according to equation (5)."""
        erf_term = erf((m * x + b) / (self.sqrt2 * self.sigma_x))
        exp_term = np.exp(-0.5 * (x / self.sigma_x) ** 2)
        return erf_term * exp_term

    def calculate_segment_probability(self, edge: OctagonEdge) -> float:
        norm_factor = 1.0 / (np.sqrt(8 * np.pi) * self.sigma_x)
        result = quad(
            lambda x: self._integrand(x, edge.m, edge.b),
            edge.xl,
            edge.xu
        )[0]
        return norm_factor * result * edge.contribution

    def calculate_csp(self, octagon: np.ndarray, obstacle_pos: np.ndarray,
                      obstacle_rot: float) -> float:
        transformed_octagon = transform_to_obstacle_centered(
            octagon, obstacle_pos, obstacle_rot)
        edges = split_octagon_segments(transformed_octagon)

        total_csp = 0.0
        for edge in edges:
            total_csp += self.calculate_segment_probability(edge)

        return max(0.0, min(1.0, total_csp))


def transform_to_obstacle_centered(points: np.ndarray, pos: np.ndarray, rot: float) -> np.ndarray:
    """Transform points to obstacle-centered coordinates."""
    # Translation
    translated = points - pos

    # Rotation matrix
    c, s = np.cos(-rot), np.sin(-rot)
    R = np.array([[c, -s], [s, c]])

    return translated @ R.T


def visualize_csp_calculation(calculator: CSPCalculator,
                            octagon: np.ndarray,
                            obstacle_pos: np.ndarray,
                            obstacle_rot: float,
                            sigma_x: float,
                            sigma_y: float,
                            grid_size: int = 50) -> None:
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)

    transformed_octagon = transform_to_obstacle_centered(octagon, obstacle_pos, obstacle_rot)
    edges = split_octagon_segments(transformed_octagon)

    # Calculate Gaussian for visualization
    Z = np.zeros_like(X)
    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j] = np.exp(-0.5 * ((X[i,j]/sigma_x)**2 + (Y[i,j]/sigma_y)**2)) / (
                2 * np.pi * sigma_x * sigma_y)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Original configuration
    ax1.set_title('Original Configuration')
    ax1.add_patch(Polygon(octagon, fill=False, color='black'))
    ax1.plot(obstacle_pos[0], obstacle_pos[1], 'ro', label='Obstacle Center')
    ax1.arrow(obstacle_pos[0], obstacle_pos[1],
              np.cos(obstacle_rot), np.sin(obstacle_rot),
              head_width=0.1, color='red')
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()

    # Transformed space
    ax2.set_title('Transformed Space with Probability Heatmap')
    im = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')

    # Plot edges with colors based on contribution
    for edge in edges:
        x_vals = np.array([edge.xl, edge.xu])
        y_vals = edge.m * x_vals + edge.b
        color = 'red' if edge.contribution > 0 else 'blue'
        ax2.plot(x_vals, y_vals, color=color, linewidth=2)

    ax2.set_aspect('equal')
    ax2.grid(True)
    plt.colorbar(im, ax=ax2, label='Probability Density')

    csp = calculator.calculate_csp(octagon, obstacle_pos, obstacle_rot)
    plt.suptitle(f'Total CSP: {csp:.4f}')

    plt.tight_layout()
    plt.show()


irregular_octagon = np.array([
    [0, 0], [3, 0], [4, 2], [3.5, 4],
    [2, 5], [-0.5, 4.5], [-2, 3], [-1.5, 1]
])

# Test configurations
obstacle_positions = [
    np.array([0.0, 2.0]),    # Near center
    np.array([2.0, 3.0]),    # Upper right
    np.array([-1.0, 1.0]),   # Lower left
    np.array([4.0, 4.0])     # Upper middle
]

obstacle_rotations = [0.0, np.pi/4, -np.pi/6]  # Mix of rotations

# Run visualization
calculator = CSPCalculator(sigma_x=1.0, sigma_y=0.5)

for pos in obstacle_positions:
    for rot in obstacle_rotations:
        visualize_csp_calculation(calculator, irregular_octagon, pos, rot, sigma_x=1.0, sigma_y=0.5)
