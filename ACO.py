

import numpy as np
import cv2
import random
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class Ellipse:
    """Class to represent an ellipse with parameters"""
    def __init__(self, center: Tuple[float, float], a: float, b: float, theta: float):
        self.center = center  # (x, y)
        self.a = a  # semi-major axis
        self.b = b  # semi-minor axis
        self.theta = theta  # rotation angle in radians

    def get_points(self, n: int = 100) -> np.ndarray:
        """Generate points on the ellipse for visualization"""
        t = np.linspace(0, 2*np.pi, n)
        x = self.center[0] + self.a * np.cos(t) * np.cos(self.theta) - self.b * np.sin(t) * np.sin(self.theta)
        y = self.center[1] + self.a * np.cos(t) * np.sin(self.theta) + self.b * np.sin(t) * np.cos(self.theta)
        return np.column_stack((x, y))

    def point_distance(self, point: Tuple[float, float]) -> float:
        """Calculate distance from a point to ellipse"""
        x, y = point
        x0, y0 = self.center

        # Translate point to ellipse coordinate system
        x_trans = x - x0
        y_trans = y - y0

        # Rotate point to align with ellipse axes
        x_rot = x_trans * np.cos(self.theta) + y_trans * np.sin(self.theta)
        y_rot = -x_trans * np.sin(self.theta) + y_trans * np.cos(self.theta)

        # Calculate algebraic distance
        return abs((x_rot**2 / self.a**2) + (y_rot**2 / self.b**2) - 1)

    def fitness(self, edge_points: np.ndarray, threshold: float = 2.0) -> float:
        """Calculate fitness based on how many edge points lie on the ellipse"""
        if len(edge_points) == 0:
            return 0

        distances = np.array([self.point_distance((x, y)) for x, y in edge_points])

        # Points within threshold are considered on ellipse
        on_ellipse = np.sum(distances < threshold)

        # Normalize by total points and ellipse size (prefer smaller ellipses)
        coverage = on_ellipse / len(edge_points)
        size_penalty = 1 / (1 + self.a + self.b)  # Penalize very large ellipses

        return coverage * size_penalty

class ACO_EllipseDetection:
    """Ant Colony Optimization for Ellipse Detection"""

    def __init__(self,
                 n_ants: int = 20,
                 n_iterations: int = 100,
                 pheromone_decay: float = 0.1,
                 alpha: float = 1.0,
                 beta: float = 2.0,
                 n_best: int = 5,
                 min_edge_points: int = 10,
                 ellipse_threshold: float = 2.0):

        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = pheromone_decay
        self.alpha = alpha
        self.beta = beta
        self.n_best = n_best
        self.min_edge_points = min_edge_points
        self.ellipse_threshold = ellipse_threshold

        # Edge points extracted from image
        self.edge_points = None

        # Pheromone matrix (on edge points)
        self.pheromone = None

        # Best ellipse found
        self.best_ellipse = None
        self.best_fitness = 0

        # History for visualization
        self.fitness_history = []

    def preprocess_image(self, image_path: str,
                        canny_low: int = 50,
                        canny_high: int = 150) -> np.ndarray:
        """Preprocess image to extract edge points"""
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 1.5)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, canny_low, canny_high)

        # Get edge points coordinates
        y_coords, x_coords = np.where(edges > 0)
        edge_points = np.column_stack((x_coords, y_coords))

        print(f"Found {len(edge_points)} edge points")
        return edge_points

    def initialize_pheromone(self, n_points: int):
        """Initialize pheromone matrix"""
        # Initialize with small constant value
        self.pheromone = np.ones(n_points) * 0.1

    def select_edge_points(self, ant_idx: int, n_points_needed: int = 5) -> np.ndarray:
        """Select edge points for an ant using probabilistic selection"""
        selected_points = []
        available_indices = list(range(len(self.edge_points)))

        # Probabilistic selection based on pheromone
        for _ in range(n_points_needed):
            if not available_indices:
                break

            probabilities = self.pheromone[available_indices] ** self.alpha
            probabilities = probabilities / probabilities.sum()

            # Select point
            selected_idx = np.random.choice(available_indices, p=probabilities)
            selected_points.append(self.edge_points[selected_idx])

            # Remove selected point from available indices
            available_indices.remove(selected_idx)

        return np.array(selected_points)

    def fit_ellipse_from_points(self, points: np.ndarray) -> Optional[Ellipse]:
        """Fit ellipse to given points using direct least squares"""
        if len(points) < 5:
            return None

        try:
            # Convert to float for calculations
            points = points.astype(np.float64)

            # Center the points
            centroid = np.mean(points, axis=0)
            centered = points - centroid

            # Build design matrix
            x = centered[:, 0]
            y = centered[:, 1]

            # Build constraint matrix
            D = np.column_stack([
                x**2, x*y, y**2,
                x, y,
                np.ones(len(points))
            ])

            # Build scatter matrix
            S = D.T @ D

            # Build constraint matrix for ellipse
            C = np.zeros((6, 6))
            C[0, 2] = 2
            C[1, 1] = -1
            C[2, 0] = 2

            # Solve generalized eigenvalue problem
            eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S) @ C)

            # Find eigenvector with positive eigenvalues
            pos_indices = np.where(eigvals > 0)[0]
            if len(pos_indices) == 0:
                return None

            # Get the eigenvector corresponding to the smallest positive eigenvalue
            best_idx = pos_indices[np.argmin(eigvals[pos_indices])]
            params = eigvecs[:, best_idx]

            # Extract ellipse parameters
            A, B, C, D, E, F = params

            # Convert to canonical form
            denominator = B**2 - 4*A*C
            if abs(denominator) < 1e-10:
                return None

            # Calculate center
            center_x = (2*C*D - B*E) / denominator
            center_y = (2*A*E - B*D) / denominator

            center = (center_x + centroid[0], center_y + centroid[1])

            # Calculate major/minor axes and rotation
            numerator = 2 * (A*E**2 + C*D**2 + F*B**2 - 2*B*D*E - A*C*F)
            denominator1 = (B**2 - 4*A*C) * (
                np.sqrt((A-C)**2 + B**2) - (A+C)
            )
            denominator2 = (B**2 - 4*A*C) * (
                -np.sqrt((A-C)**2 + B**2) - (A+C)
            )

            if denominator1 <= 0 or denominator2 <= 0:
                return None

            a = np.sqrt(numerator / denominator1)
            b = np.sqrt(numerator / denominator2)

            # Ensure a is major axis
            if a < b:
                a, b = b, a
                theta = 0.5 * np.arctan2(B, A-C) + np.pi/2
            else:
                theta = 0.5 * np.arctan2(B, A-C)

            # Normalize angle to [0, pi]
            theta = theta % np.pi

            # Reject invalid ellipses
            if a < 5 or b < 5 or a > 1000 or b > 1000:
                return None

            return Ellipse(center, a, b, theta)

        except Exception as e:
            # Fallback to simple fitting if direct method fails
            return self.simple_ellipse_fitting(points)

    def simple_ellipse_fitting(self, points: np.ndarray) -> Optional[Ellipse]:
        """Simple ellipse fitting using moment-based method"""
        if len(points) < 5:
            return None

        # Calculate centroid
        centroid = np.mean(points, axis=0)

        # Center points
        centered = points - centroid

        # Calculate second moments
        xx = np.mean(centered[:, 0]**2)
        yy = np.mean(centered[:, 1]**2)
        xy = np.mean(centered[:, 0] * centered[:, 1])

        # Calculate orientation
        theta = 0.5 * np.arctan2(2*xy, xx - yy)

        # Rotate points to align with axes
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_rot = centered[:, 0] * cos_theta + centered[:, 1] * sin_theta
        y_rot = -centered[:, 0] * sin_theta + centered[:, 1] * cos_theta

        # Calculate axes lengths
        a = 2 * np.sqrt(np.mean(x_rot**2))
        b = 2 * np.sqrt(np.mean(y_rot**2))

        # Ensure a is major axis
        if a < b:
            a, b = b, a
            theta += np.pi/2

        # Normalize angle
        theta = theta % np.pi

        # Create ellipse
        return Ellipse(centroid, a, b, theta)

    def update_pheromone(self, ellipses: List[Ellipse], fitness_values: List[float]):
        """Update pheromone levels based on ellipse fitness"""
        # Evaporate pheromone
        self.pheromone *= (1 - self.decay)

        # Sort ellipses by fitness
        sorted_indices = np.argsort(fitness_values)[::-1]

        # Deposit pheromone from best ants
        for idx in sorted_indices[:self.n_best]:
            if fitness_values[idx] > 0:
                ellipse = ellipses[idx]

                # Calculate pheromone deposit
                deposit = fitness_values[idx] * 10

                # Find edge points close to this ellipse
                for i, point in enumerate(self.edge_points):
                    dist = ellipse.point_distance(point)
                    if dist < self.ellipse_threshold:
                        self.pheromone[i] += deposit / (1 + dist)

    def detect(self, image_path: str, n_ellipses: int = 3) -> List[Ellipse]:
        """Main detection function"""
        # Preprocess image
        self.edge_points = self.preprocess_image(image_path)

        if len(self.edge_points) < self.min_edge_points:
            print("Not enough edge points found!")
            return []

        # Initialize pheromone
        self.initialize_pheromone(len(self.edge_points))

        detected_ellipses = []

        # Detect multiple ellipses
        for ellipse_num in range(n_ellipses):
            print(f"\nDetecting ellipse {ellipse_num + 1}/{n_ellipses}")

            self.best_ellipse = None
            self.best_fitness = 0
            self.fitness_history = []

            # ACO main loop
            for iteration in range(self.n_iterations):
                ellipses = []
                fitness_values = []

                # Each ant constructs a solution
                for ant in range(self.n_ants):
                    # Select edge points for this ant
                    selected_points = self.select_edge_points(ant, n_points_needed=10)

                    # Fit ellipse to selected points
                    ellipse = self.fit_ellipse_from_points(selected_points)

                    if ellipse is not None:
                        # Calculate fitness
                        fitness = ellipse.fitness(self.edge_points, self.ellipse_threshold)

                        ellipses.append(ellipse)
                        fitness_values.append(fitness)

                        # Update best ellipse
                        if fitness > self.best_fitness:
                            self.best_fitness = fitness
                            self.best_ellipse = ellipse
                    else:
                        ellipses.append(None)
                        fitness_values.append(0)

                # Update pheromone
                valid_ellipses = [e for e in ellipses if e is not None]
                valid_fitness = [f for e, f in zip(ellipses, fitness_values) if e is not None]

                if valid_ellipses:
                    self.update_pheromone(valid_ellipses, valid_fitness)

                # Track fitness history
                self.fitness_history.append(self.best_fitness)

                if iteration % 20 == 0:
                    print(f"Iteration {iteration}: Best Fitness = {self.best_fitness:.4f}")

            if self.best_ellipse and self.best_fitness > 0.1:
                detected_ellipses.append(self.best_ellipse)

                # Remove edge points belonging to detected ellipse
                self.remove_detected_points(self.best_ellipse)
            else:
                break

        return detected_ellipses

    def remove_detected_points(self, ellipse: Ellipse):
        """Remove edge points that belong to detected ellipse"""
        indices_to_remove = []

        for i, point in enumerate(self.edge_points):
            if ellipse.point_distance(point) < self.ellipse_threshold:
                indices_to_remove.append(i)

        # Remove points
        self.edge_points = np.delete(self.edge_points, indices_to_remove, axis=0)

        # Update pheromone
        if len(self.pheromone) > len(indices_to_remove):
            self.pheromone = np.delete(self.pheromone, indices_to_remove)

        print(f"Removed {len(indices_to_remove)} points belonging to detected ellipse")

    def visualize_results(self, original_image_path: str, ellipses: List[Ellipse],
                         save_path: Optional[str] = None):
        """Visualize detected ellipses"""
        # Read original image
        if isinstance(original_image_path, str):
            original_image = cv2.imread(original_image_path)
        else:
            original_image = cv2.cvtColor(original_image_path, cv2.COLOR_GRAY2BGR)

        # Draw detected ellipses
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i, ellipse in enumerate(ellipses):
            color = colors[i % len(colors)]

            # Convert ellipse parameters to OpenCV format
            center = (int(ellipse.center[0]), int(ellipse.center[1]))
            axes = (int(ellipse.a), int(ellipse.b))
            angle = np.degrees(ellipse.theta)

            # Draw ellipse
            cv2.ellipse(original_image, center, axes, angle, 0, 360, color, 2)

            # Draw center
            cv2.circle(original_image, center, 5, color, -1)

            # Add label
            cv2.putText(original_image, f"Ellipse {i+1}",
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Detected Ellipses")
        axes[0].axis('off')

        # Plot edge points
        if self.edge_points is not None:
            axes[1].scatter(self.edge_points[:, 0], -self.edge_points[:, 1],
                           s=1, c='black', alpha=0.5)
            axes[1].set_title("Edge Points")
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")
            axes[1].set_aspect('equal')
            axes[1].invert_yaxis()

        # Plot convergence
        axes[2].plot(self.fitness_history, 'b-', linewidth=2)
        axes[2].set_title("Convergence History")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Best Fitness")
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

# Example usage and synthetic image generation
def generate_test_image(size=(400, 400), n_ellipses=2):
    """Generate a test image with ellipses"""
    # Create blank image
    image = np.zeros(size, dtype=np.uint8)

    # Add random ellipses
    for _ in range(n_ellipses):
        center = (np.random.randint(100, size[1]-100),
                 np.random.randint(100, size[0]-100))
        axes = (np.random.randint(40, 100), np.random.randint(20, 80))
        angle = np.random.randint(0, 180)

        # Draw ellipse
        cv2.ellipse(image, center, axes, angle, 0, 360, 255, 2)

    # Add some noise
    noise = np.random.normal(0, 30, size).astype(np.uint8)
    image = cv2.add(image, noise)

    # Threshold
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    return image

# Main execution
if __name__ == "__main__":
    # Option 1: Generate synthetic test image
    print("Generating synthetic test image...")
    test_image = generate_test_image(size=(400, 400), n_ellipses=2)

    # Option 2: Use real image (uncomment and provide path)
    # test_image = "path/to/your/image.jpg"

    # Initialize ACO Ellipse Detector
    detector = ACO_EllipseDetection(
        n_ants=15,
        n_iterations=80,
        pheromone_decay=0.1,
        alpha=1.0,
        beta=2.0,
        n_best=3,
        min_edge_points=20,
        ellipse_threshold=2.0
    )

    # Detect ellipses
    print("\nStarting ellipse detection with ACO...")
    detected_ellipses = detector.detect(test_image, n_ellipses=2)

    # Display results
    print(f"\nDetected {len(detected_ellipses)} ellipses:")
    for i, ellipse in enumerate(detected_ellipses):
        print(f"Ellipse {i+1}:")
        print(f"  Center: ({ellipse.center[0]:.1f}, {ellipse.center[1]:.1f})")
        print(f"  Axes: {ellipse.a:.1f} x {ellipse.b:.1f}")
        print(f"  Angle: {np.degrees(ellipse.theta):.1f}°")
        print(f"  Fitness: {ellipse.fitness(detector.edge_points):.4f}")

    # Visualize results
    detector.visualize_results(test_image, detected_ellipses)