import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LowLevelPolicy(nn.Module):
    """
    Low-level policy that converts waypoints into control actions.
    This policy determines the actual control commands for the drone.
    """
    
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(LowLevelPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.device = device
        
        # Input dimension: state + waypoint
        input_dim = state_dim + 3  # Waypoint is 3D position
        
        # Policy network (outputs actions)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Control actions
        )
        
        # Value network for critic
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value estimate
        )
        
        # Gaussian policy parameters
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, input_data, image=None, deterministic=False):
        """
        Forward pass to compute the low-level control actions.
        
        Args:
            input_data: Combined state and waypoint
            image: Image from the drone (not used at this level)
            deterministic: Whether to use deterministic action selection
        
        Returns:
            action: Control action for the drone
        """
        # Get policy output (mean of Gaussian policy)
        mean = self.policy_net(input_data)
        mean = torch.tanh(mean)  # Bound actions to [-1, 1]
        
        if deterministic:
            # Return mean for deterministic action
            return mean
        else:
            # Sample from Gaussian for exploration
            std = torch.exp(self.log_std)
            normal = torch.randn_like(mean) * std + mean
            
            # Bound actions to [-1, 1]
            action = torch.clamp(normal, -1.0, 1.0)
            
            return action
    
    def evaluate_actions(self, inputs, images, actions):
        """
        Evaluate actions for training.
        
        Args:
            inputs: Batch of combined state and waypoint
            images: Batch of images (not used)
            actions: Actions to evaluate
        
        Returns:
            values: Value estimates
            action_log_probs: Log probabilities of actions
            dist_entropy: Entropy of the policy distribution
        """
        # Get policy output and value
        mean = self.policy_net(inputs)
        mean = torch.tanh(mean)  # Bound actions to [-1, 1]
        values = self.value_net(inputs)
        
        # Compute log probabilities and entropy
        std = torch.exp(self.log_std)
        
        action_log_probs = -0.5 * (((actions - mean) / std).pow(2) + 2 * self.log_std + np.log(2 * np.pi))
        action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
        
        dist_entropy = 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std
        dist_entropy = dist_entropy.sum(-1).mean()
        
        return values, action_log_probs, dist_entropy
