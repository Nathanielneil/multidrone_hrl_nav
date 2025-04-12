import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import io
import cv2
from mpl_toolkits.mplot3d import Axes3D
import time
import torch

class TrainingVisualizer:
    """Visualizer for training metrics."""
    
    def __init__(self, config):
        """
        Initialize the training visualizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Create visualization directory
        self.vis_dir = os.path.join(config.log_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Metrics to track
        self.metrics = {
            'rewards': [],
            'lengths': [],
            'losses': [],
            'success_rates': [],
            'distances': []
        }
        
        # Initialize figure
        self.fig, self.axs = plt.subplots(3, 2, figsize=(15, 12))
        self.fig.tight_layout(pad=3.0)
    
    def update_metrics(self, metrics):
        """
        Update metrics for visualization.
        
        Args:
            metrics: Dictionary of metrics to update
        """
        if 'reward' in metrics or 'mean_reward' in metrics:
            self.metrics['rewards'].append(metrics.get('reward', metrics.get('mean_reward')))
        
        if 'length' in metrics or 'mean_length' in metrics:
            self.metrics['lengths'].append(metrics.get('length', metrics.get('mean_length')))
        
        if 'loss' in metrics or 'mean_loss' in metrics:
            self.metrics['losses'].append(metrics.get('loss', metrics.get('mean_loss')))
        
        if 'success_rate' in metrics:
            self.metrics['success_rates'].append(metrics['success_rate'])
        
        if 'distance' in metrics or 'mean_distance' in metrics:
            self.metrics['distances'].append(metrics.get('distance', metrics.get('mean_distance')))
    
    def plot_metrics(self, save=True):
        """
        Plot training metrics.
        
        Args:
            save: Whether to save the plot
        """
        # Clear axes
        for ax in self.axs.flat:
            ax.clear()
        
        # Plot rewards
        if self.metrics['rewards']:
            self.axs[0, 0].plot(self.metrics['rewards'])
            self.axs[0, 0].set_title('Average Reward')
            self.axs[0, 0].set_xlabel('Episode')
            self.axs[0, 0].set_ylabel('Reward')
            self.axs[0, 0].grid(True)
        
        # Plot episode lengths
        if self.metrics['lengths']:
            self.axs[0, 1].plot(self.metrics['lengths'])
            self.axs[0, 1].set_title('Episode Length')
            self.axs[0, 1].set_xlabel('Episode')
            self.axs[0, 1].set_ylabel('Steps')
            self.axs[0, 1].grid(True)
        
        # Plot losses
        if self.metrics['losses']:
            self.axs[1, 0].plot(self.metrics['losses'])
            self.axs[1, 0].set_title('Loss')
            self.axs[1, 0].set_xlabel('Update')
            self.axs[1, 0].set_ylabel('Loss')
            self.axs[1, 0].grid(True)
        
        # Plot success rates
        if self.metrics['success_rates']:
            self.axs[1, 1].plot(self.metrics['success_rates'])
            self.axs[1, 1].set_title('Success Rate')
            self.axs[1, 1].set_xlabel('Episode')
            self.axs[1, 1].set_ylabel('Success Rate')
            self.axs[1, 1].set_ylim([0, 1])
            self.axs[1, 1].grid(True)
        
        # Plot distances
        if self.metrics['distances']:
            self.axs[2, 0].plot(self.metrics['distances'])
            self.axs[2, 0].set_title('Distance to Target')
            self.axs[2, 0].set_xlabel('Episode')
            self.axs[2, 0].set_ylabel('Distance')
            self.axs[2, 0].grid(True)
        
        # Plot moving averages
        if len(self.metrics['rewards']) > 10:
            window_size = min(10, len(self.metrics['rewards']))
            rewards_avg = np.convolve(
                self.metrics['rewards'], 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            self.axs[2, 1].plot(rewards_avg)
            self.axs[2, 1].set_title('Smoothed Reward (MA-10)')
            self.axs[2, 1].set_xlabel('Episode')
            self.axs[2, 1].set_ylabel('Reward')
            self.axs[2, 1].grid(True)
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Save figure if requested
        if save:
            plt.savefig(os.path.join(self.vis_dir, 'training_metrics.png'), dpi=200)
    
    def create_animation(self, trajectories, save_path='trajectory_animation.mp4'):
        """
        Create an animation of drone trajectories.
        
        Args:
            trajectories: List of trajectories for each drone
            save_path: Path to save the animation
        """
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Trajectories')
        
        # Set limits
        max_x = max([max([p[0] for p in traj]) for traj in trajectories])
        max_y = max([max([p[1] for p in traj]) for traj in trajectories])
        max_z = max([max([p[2] for p in traj]) for traj in trajectories])
        min_x = min([min([p[0] for p in traj]) for traj in trajectories])
        min_y = min([min([p[1] for p in traj]) for traj in trajectories])
        min_z = min([min([p[2] for p in traj]) for traj in trajectories])
        
        ax.set_xlim([min_x - 5, max_x + 5])
        ax.set_ylim([min_y - 5, max_y + 5])
        ax.set_zlim([min_z - 5, max_z + 5])
        
        # Colors for drones
        colors = ['b', 'g', 'm']
        
        # Plot target
        target = self.config.target_pos
        ax.scatter(target[0], target[1], target[2], c='r', marker='*', s=100, label='Target')
        
        # Initialize drone plots
        drone_plots = []
        for i in range(len(trajectories)):
            drone, = ax.plot([], [], [], 'o-', c=colors[i % len(colors)], markersize=8, label=f'Drone {i+1}')
            drone_plots.append(drone)
        
        # Add legend
        ax.legend()
        
        # Get maximum trajectory length
        max_len = max([len(traj) for traj in trajectories])
        
        # Animation function
        def animate(i):
            for d, drone in enumerate(drone_plots):
                if i < len(trajectories[d]):
                    x_data = [p[0] for p in trajectories[d][:i+1]]
                    y_data = [p[1] for p in trajectories[d][:i+1]]
                    z_data = [p[2] for p in trajectories[d][:i+1]]
                    drone.set_data(x_data, y_data)
                    drone.set_3d_properties(z_data)
            
            return drone_plots
        
        # Create animation
        anim = FuncAnimation(
            fig, animate, frames=max_len,
            interval=100, blit=True
        )
        
        # Save animation
        anim.save(os.path.join(self.vis_dir, save_path), writer='ffmpeg', fps=10, dpi=200)
        
        plt.close()
    
    def plot_drone_positions(self, positions, save_path='drone_positions.png'):
        """
        Plot drone positions.
        
        Args:
            positions: List of positions for each drone
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Positions')
        
        # Colors for drones
        colors = ['b', 'g', 'm']
        
        # Plot drone positions
        for i, pos in enumerate(positions):
            ax.scatter(
                pos[0], pos[1], pos[2],
                c=colors[i % len(colors)],
                marker='o',
                s=100,
                label=f'Drone {i+1}'
            )
        
        # Plot target
        target = self.config.target_pos
        ax.scatter(
            target[0], target[1], target[2],
            c='r',
            marker='*',
            s=200,
            label='Target'
        )
        
        # Add legend
        ax.legend()
        
        # Save figure
        plt.savefig(os.path.join(self.vis_dir, save_path), dpi=200)
        plt.close()
    
    def plot_attention_heatmap(self, attention_weights, drone_index=0, step=0, save_path=None):
        """
        Plot attention heatmap for a drone.
        
        Args:
            attention_weights: Attention weights matrix
            drone_index: Index of the drone
            step: Current step
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 6))
        
        # Plot heatmap
        plt.imshow(attention_weights, cmap='viridis')
        
        # Add colorbar
        plt.colorbar()
        
        # Set title and labels
        plt.title(f'Attention Weights - Drone {drone_index+1} - Step {step}')
        plt.xlabel('Target')
        plt.ylabel('Source')
        
        # Add grid
        plt.grid(False)
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=200)
            plt.close()
        else:
            plt.show()
    
    def plot_image_with_trajectory(self, image, trajectory, drone_index=0, step=0, save_path=None):
        """
        Plot drone camera image with trajectory overlay.
        
        Args:
            image: Camera image
            trajectory: Trajectory points
            drone_index: Index of the drone
            step: Current step
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Plot image
        plt.imshow(image)
        
        # Plot trajectory points
        if trajectory:
            x_points = [p[0] for p in trajectory]
            y_points = [p[1] for p in trajectory]
            plt.plot(x_points, y_points, 'r-', linewidth=2)
            plt.scatter(x_points[-1], y_points[-1], c='r', marker='o', s=50)
        
        # Set title
        plt.title(f'Camera View with Trajectory - Drone {drone_index+1} - Step {step}')
        
        # Remove axis
        plt.axis('off')
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, dpi=200)
            plt.close()
        else:
            plt.show()
