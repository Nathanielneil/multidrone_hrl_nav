import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MidLevelPolicy(nn.Module):
    """
    Mid-level policy that converts high-level goals into waypoints.
    This policy determines intermediate waypoints for each drone to follow.
    """
    
    def __init__(self, state_dim, hidden_dim, goal_dim, device):
        super(MidLevelPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.goal_dim = goal_dim
        self.device = device
        
        # Input dimension: state + high-level goal
        input_dim = state_dim + goal_dim
        
        # Policy network (outputs waypoints)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim)  # 3D waypoint position
        )
        
        # Value network for critic
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value estimate
        )
        
        # Gaussian policy parameters
        self.log_std = nn.Parameter(torch.zeros(goal_dim))
    
    def forward(self, input_data, image=None, deterministic=False):
        """
        Forward pass to compute the mid-level waypoints.
        
        Args:
            input_data: Combined state and high-level goal
            image: Image from the drone (not used at this level)
            deterministic: Whether to use deterministic action selection
        
        Returns:
            waypoint: Waypoint position for the drone
        """
        # Get policy output (mean of Gaussian policy)
        mean = self.policy_net(input_data)
        
        if deterministic:
            # Return mean for deterministic action
            return mean
        else:
            # Sample from Gaussian for exploration
            std = torch.exp(self.log_std)
            normal = torch.randn_like(mean) * std + mean
            
            return normal
    
    def evaluate_actions(self, inputs, images, actions):
        """
        Evaluate actions for training.
        
        Args:
            inputs: Batch of combined state and high-level goal
            images: Batch of images (not used)
            actions: Actions to evaluate
        
        Returns:
            values: Value estimates
            action_log_probs: Log probabilities of actions
            dist_entropy: Entropy of the policy distribution
        """
        # Get policy output and value
        mean = self.policy_net(inputs)
        values = self.value_net(inputs)
        
        # Compute log probabilities and entropy
        std = torch.exp(self.log_std)
        
        action_log_probs = -0.5 * (((actions - mean) / std).pow(2) + 2 * self.log_std + np.log(2 * np.pi))
        action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
        
        dist_entropy = 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std
        dist_entropy = dist_entropy.sum(-1).mean()
        
        return values, action_log_probs, dist_entropy
