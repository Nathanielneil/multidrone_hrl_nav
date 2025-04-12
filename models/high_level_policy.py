import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNFeatureExtractor(nn.Module):
    """CNN for extracting features from images."""
    
    def __init__(self, input_channels=3, output_dim=512):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate output dimension of the convolutional layers
        # Assuming input image size is 84x84
        conv_output_size = self._get_conv_output_size(input_channels, 84, 84)
        
        self.fc = nn.Linear(conv_output_size, output_dim)
    
    def _get_conv_output_size(self, input_channels, height, width):
        # Create a dummy input to calculate the output size
        dummy_input = torch.zeros(1, input_channels, height, width)
        
        x = F.relu(self.conv1(dummy_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        return int(np.prod(x.size()))
    
    def forward(self, x):
        # Input: [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layer
        x = F.relu(self.fc(x))
        
        return x

class HighLevelPolicy(nn.Module):
    """
    High-level policy that determines global goals for each drone.
    This policy takes the global state and assigns goals to each drone.
    """
    
    def __init__(self, state_dim, hidden_dim, num_drones, cnn_features_dim, device):
        super(HighLevelPolicy, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_drones = num_drones
        self.device = device
        self.use_images = True
        
        # CNN for processing images
        self.cnn = CNNFeatureExtractor(input_channels=3, output_dim=cnn_features_dim)
        
        # MLP for processing state information
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combine CNN features with state features
        combined_dim = hidden_dim + cnn_features_dim
        
        # Policy network (outputs goals for each drone)
        self.policy_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_drones * 3)  # 3D goal position for each drone
        )
        
        # Value network for critic
        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value estimate
        )
        
        # Gaussian policy parameters
        self.log_std = nn.Parameter(torch.zeros(num_drones * 3))
    
    def forward(self, state, image=None, deterministic=False):
        """
        Forward pass to compute the high-level goals.
        
        Args:
            state: Global state tensor
            image: Stack of images from all drones (optional)
            deterministic: Whether to use deterministic action selection
        
        Returns:
            goals: List of goal positions for each drone
        """
        # Process state information
        state_features = self.state_encoder(state)
        
        # Process images if available
        if image is not None and self.use_images:
            if image.dim() == 3:  # Single image
                image = image.unsqueeze(0)  # Add batch dimension
            
            if image.dim() == 4:  # ÐÎ×´Îª [num_drones, channels, height, width]  # Multiple images [num_drones, channels, height, width]
                # Reshape to [batch_size, channels, height, width]
                batch_size, channels, height, width = image.shape
                image = image.view(batch_size, channels, height, width)
                
                # Process images through CNN
                image_features = self.cnn(image)
                
                # Average features across drones
                image_features = image_features.mean(dim=0)
            else:
                # Create dummy image features
                image_features = torch.zeros(self.cnn.fc.out_features, device=self.device)
        else:
            # Create dummy image features
            image_features = torch.zeros(self.cnn.fc.out_features, device=self.device)
        
        # Combine features
        combined_features = torch.cat([state_features, image_features])
        
        # Get policy output (mean of Gaussian policy)
        mean = self.policy_net(combined_features)
        
        # Reshape to [num_drones, 3]
        mean = mean.view(self.num_drones, 3)
        
        if deterministic:
            # Return mean for deterministic action
            return [mean[i] for i in range(self.num_drones)]
        else:
            # Sample from Gaussian for exploration
            std = torch.exp(self.log_std).view(self.num_drones, 3)
            normal = torch.randn_like(mean) * std + mean
            
            return [normal[i] for i in range(self.num_drones)]
    
    def evaluate_actions(self, state, image, actions):
        """
        Evaluate actions for training.
        
        Args:
            state: Global state tensor
            image: Stack of images from all drones
            actions: Actions to evaluate
        
        Returns:
            values: Value estimates
            action_log_probs: Log probabilities of actions
            dist_entropy: Entropy of the policy distribution
        """
        # Process state and image
        state_features = self.state_encoder(state)
        
        if image is not None and self.use_images:
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            batch_size, channels, height, width = image.shape
            image_features = self.cnn(image).mean(dim=0)
        else:
            image_features = torch.zeros(self.cnn.fc.out_features, device=self.device)
        
        combined_features = torch.cat([state_features, image_features])
        
        # Get policy output and value
        mean = self.policy_net(combined_features)
        values = self.value_net(combined_features)
        
        # Compute log probabilities and entropy
        std = torch.exp(self.log_std)
        actions = actions.view(-1, self.num_drones * 3)
        
        action_log_probs = -0.5 * (((actions - mean) / std).pow(2) + 2 * self.log_std + np.log(2 * np.pi))
        action_log_probs = action_log_probs.sum(dim=-1, keepdim=True)
        
        dist_entropy = 0.5 + 0.5 * np.log(2 * np.pi) + self.log_std
        dist_entropy = dist_entropy.sum(-1).mean()
        
        return values, action_log_probs, dist_entropy
