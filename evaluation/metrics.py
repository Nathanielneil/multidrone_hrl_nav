import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

class MetricsCollector:
    """Collector for evaluation metrics."""
    
    def __init__(self, config):
        """
        Initialize the metrics collector.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        
        # Create metrics directory
        self.metrics_dir = os.path.join(config.log_dir, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
    
    def add_episode_metric(self, name, value):
        """Add an episode-level metric."""
        self.episode_metrics[name].append(value)
    
    def add_step_metric(self, name, value):
        """Add a step-level metric."""
        self.step_metrics[name].append(value)
    
    def add_metric(self, name, value):
        """Add a general metric."""
        self.metrics[name].append(value)
    
    def compute_episode_metrics(self, rewards, lengths, distances, success):
        """
        Compute episode-level metrics.
        
        Args:
            rewards: List of episode rewards
            lengths: List of episode lengths
            distances: List of final distances to target
            success: Boolean indicating whether the episode was successful
        
        Returns:
            Dictionary of computed metrics
        """
        metrics = {
            'episode_reward': np.mean(rewards),
            'episode_length': np.mean(lengths),
            'final_distance': np.mean(distances),
            'success_rate': float(success)
        }
        
        # Add to episode metrics
        for key, value in metrics.items():
            self.add_episode_metric(key, value)
        
        return metrics
    
    def compute_training_metrics(self):
        """
        Compute metrics over the entire training.
        
        Returns:
            Dictionary of computed metrics
        """
        training_metrics = {}
        
        # Episode metrics
        for key, values in self.episode_metrics.items():
            training_metrics[f'mean_{key}'] = np.mean(values)
            training_metrics[f'std_{key}'] = np.std(values)
            training_metrics[f'max_{key}'] = np.max(values)
            training_metrics[f'min_{key}'] = np.min(values)
        
        # Step metrics
        for key, values in self.step_metrics.items():
            training_metrics[f'mean_{key}'] = np.mean(values)
            training_metrics[f'std_{key}'] = np.std(values)
        
        # Other metrics
        for key, values in self.metrics.items():
            training_metrics[f'mean_{key}'] = np.mean(values)
            training_metrics[f'std_{key}'] = np.std(values)
        
        return training_metrics
    
    def save_metrics(self, filename='metrics.json'):
        """Save metrics to file."""
        metrics = self.compute_training_metrics()
        
        # Save as JSON
        json_path = os.path.join(self.metrics_dir, filename)
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {json_path}")
        
        return metrics
    
    def plot_metrics(self, save_dir=None):
        """
        Plot metrics.
        
        Args:
            save_dir: Directory to save plots
        """
        if save_dir is None:
            save_dir = self.metrics_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot episode metrics
        for key, values in self.episode_metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(f'Episode {key}')
            plt.xlabel('Episode')
            plt.ylabel(key)
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'episode_{key}.png'))
            plt.close()
        
        # Plot step metrics
        for key, values in self.step_metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(f'Step {key}')
            plt.xlabel('Step')
            plt.ylabel(key)
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'step_{key}.png'))
            plt.close()
        
        # Plot other metrics
        for key, values in self.metrics.items():
            plt.figure(figsize=(10, 6))
            plt.plot(values)
            plt.title(key)
            plt.xlabel('Iteration')
            plt.ylabel(key)
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, f'{key}.png'))
            plt.close()
        
        print(f"Plots saved to {save_dir}")
