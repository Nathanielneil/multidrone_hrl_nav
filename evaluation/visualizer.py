import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import cv2

class Visualizer:
    """Visualizer for the multi-drone environment."""
    
    def __init__(self, config, env, agent):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration object
            env: Environment to visualize
            agent: Agent to visualize
        """
        self.config = config
        self.env = env
        self.agent = agent
        
        # Create visualization directory
        self.vis_dir = os.path.join(config.log_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def visualize_episode(self, save_path=None, max_steps=None):
        """
        Visualize an episode.
        
        Args:
            save_path: Path to save the visualization
            max_steps: Maximum number of steps to visualize
        """
        # Reset environment
        obs = self.env.reset()
        done = False
        
        # Initialize lists to store trajectory
        trajectories = [[] for _ in range(self.env.num_drones)]
        rewards = []
        
        # Maximum number of steps
        if max_steps is None:
            max_steps = self.env.max_steps_per_episode
        
        # Initialize 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Multi-Drone Exploration')
        
        # Set axis limits
        ax.set_xlim([-10, 60])
        ax.set_ylim([-10, 60])
        ax.set_zlim([-20, 0])
        
        # Plot target
        target = self.config.target_pos
        ax.scatter(target[0], target[1], target[2], c='r', marker='*', s=200, label='Target')
        
        # Add target sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = target[0] + self.config.target_radius * np.outer(np.cos(u), np.sin(v))
        y = target[1] + self.config.target_radius * np.outer(np.sin(u), np.sin(v))
        z = target[2] + self.config.target_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='r', alpha=0.1)
        
        # Run episode
        step = 0
        
        # Create video writer if save_path is provided
        if save_path is not None:
            if not save_path.endswith('.mp4'):
                save_path += '.mp4'
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.vis_dir, save_path)
            fps = 10
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (fig.get_figwidth() * 100, fig.get_figheight() * 100))
        
        # Drone colors
        colors = ['b', 'g', 'm']
        
        # Initialize drone plots
        drone_plots = []
        for i in range(self.env.num_drones):
            drone_plot, = ax.plot([], [], [], 'o-', c=colors[i], markersize=8, label=f'Drone {i+1}')
            drone_plots.append(drone_plot)
        
        # Add legend
        ax.legend()
        
        # Update plot function
        def update_plot():
            for i in range(self.env.num_drones):
                if trajectories[i]:
                    x_data = [pos[0] for pos in trajectories[i]]
                    y_data = [pos[1] for pos in trajectories[i]]
                    z_data = [pos[2] for pos in trajectories[i]]
                    drone_plots[i].set_data(x_data, y_data)
                    drone_plots[i].set_3d_properties(z_data)
            
            fig.canvas.draw()
            
            # Convert plot to image
            if save_path is not None:
                # Convert figure to RGB image
                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                # Convert RGB to BGR for OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # Write to video
                video_writer.write(img)
        
        while not done and step < max_steps:
            # Get action from agent
            action = self.agent.act(obs, deterministic=True)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Extract drone positions
            for i, pos in enumerate(info['drone_positions']):
                trajectories[i].append([pos.x_val, pos.y_val, pos.z_val])
            
            # Append reward
            rewards.append(reward)
            
            # Update observation
            obs = next_obs
            
            # Update plot
            update_plot()
            
            # Increment step counter
            step += 1
        
        # Close video writer
        if save_path is not None:
            video_writer.release()
            print(f"Visualization saved to {video_path}")
        
        # Display final plot
        plt.tight_layout()
        plt.show()
        
        # Return metrics
        return {
            'total_reward': sum(rewards),
            'episode_length': step,
            'final_distances': info['distances_to_target']
        }
    
    def visualize_trajectories(self, trajectories, save_path=None):
        """
        Visualize drone trajectories.
        
        Args:
            trajectories: List of trajectories for each drone
            save_path: Path to save the visualization
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Drone Trajectories')
        
        # Drone colors
        colors = ['b', 'g', 'm']
        
        # Plot trajectories
        for i, trajectory in enumerate(trajectories):
            x_data = [pos[0] for pos in trajectory]
            y_data = [pos[1] for pos in trajectory]
            z_data = [pos[2] for pos in trajectory]
            
            ax.plot3D(x_data, y_data, z_data, c=colors[i], label=f'Drone {i+1}')
            
            # Mark start and end points
            ax.scatter(x_data[0], y_data[0], z_data[0], c=colors[i], marker='o', s=100)
            ax.scatter(x_data[-1], y_data[-1], z_data[-1], c=colors[i], marker='x', s=100)
        
        # Plot target
        target = self.config.target_pos
        ax.scatter(target[0], target[1], target[2], c='r', marker='*', s=200, label='Target')
        
        # Add legend
        ax.legend()
        
        # Save figure if requested
        if save_path is not None:
            if not save_path.endswith('.png'):
                save_path += '.png'
            
            plt.savefig(os.path.join(self.vis_dir, save_path), dpi=300, bbox_inches='tight')
            print(f"Trajectories saved to {os.path.join(self.vis_dir, save_path)}")
        
        # Display plot
        plt.tight_layout()
        plt.show()
    
    def visualize_camera_feed(self, drone_index=0):
        """
        Visualize camera feed from a drone.
        
        Args:
            drone_index: Index of the drone to visualize
        """
        # Reset environment
        obs = self.env.reset()
        done = False
        
        plt.figure(figsize=(8, 6))
        plt.ion()  # Turn on interactive mode
        
        # Create subplot for camera feed
        ax = plt.subplot(1, 1, 1)
        ax.set_title(f'Camera Feed - Drone {drone_index+1}')
        
        # Initial plot
        camera_img = obs['images'][drone_index]
        img_plot = ax.imshow(camera_img)
        
        plt.tight_layout()
        plt.show()
        
        # Maximum number of steps
        max_steps = self.env.max_steps_per_episode
        
        step = 0
        while not done and step < max_steps:
            # Get action from agent
            action = self.agent.act(obs, deterministic=True)
            
            # Step environment
            next_obs, reward, done, info = self.env.step(action)
            
            # Update camera feed
            camera_img = next_obs['images'][drone_index]
            img_plot.set_data(camera_img)
            
            plt.pause(0.01)
            
            # Update observation
            obs = next_obs
            
            # Increment step counter
            step += 1
        
        plt.ioff()  # Turn off interactive mode
        plt.show()
