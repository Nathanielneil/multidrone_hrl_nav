import os
import torch
import numpy as np
from .high_level_policy import HighLevelPolicy
from .mid_level_policy import MidLevelPolicy
from .low_level_policy import LowLevelPolicy
import torch.nn.functional as F

class HierarchicalAgent:
    """Hierarchical agent that combines high, mid, and low-level policies."""
    
    def __init__(self, config, device):
        """
        Initialize the hierarchical agent.
        
        Args:
            config: Configuration object
            device: Torch device (CPU or GPU)
        """
        self.config = config
        self.device = device
        self.num_drones = config.num_drones
        
        # Initialize policies
        self.high_level_policy = HighLevelPolicy(
            config.observation_space_dim * config.num_drones,
            config.high_level_hidden_dim,
            config.num_drones,
            config.cnn_features_dim,
            device
        ).to(device)
        
        self.mid_level_policy = MidLevelPolicy(
            config.observation_space_dim,
            config.mid_level_hidden_dim,
            3,  # x, y, z goal position
            device
        ).to(device)
        
        self.low_level_policy = LowLevelPolicy(
            config.observation_space_dim,
            config.low_level_hidden_dim,
            config.action_space_dim,
            device
        ).to(device)
        
        # Policy update intervals
        self.high_level_interval = config.high_level_action_interval
        self.mid_level_interval = config.mid_level_action_interval
        
        # Memory for storing goals
        self.high_level_goals = [None] * self.num_drones
        self.mid_level_goals = [None] * self.num_drones
        
        # Step counters
        self.high_level_steps = 0
        self.mid_level_steps = 0
    
    def act(self, observation, deterministic=False):
        """
        # ������Ϣ - ���ά��
        print(f"State shape: {observation[\'states\'].shape}")
        print(f"Expected state dim: {self.config.observation_space_dim * self.num_drones}")
        
        Select actions for all drones based on hierarchical policies.
        
        Args:
            observation: Combined observation from environment
            deterministic: Whether to use deterministic action selection
        
        Returns:
            actions: Combined actions for all drones
        """
        # Increment step counters
        self.high_level_steps += 1
        self.mid_level_steps += 1
        
        # Extract state and image observations
        state_obs = torch.FloatTensor(observation["states"]).to(self.device)
        image_obs = torch.FloatTensor(observation["images"]).to(self.device)
        
        # Split state observations for individual drones
        single_obs_dim = self.config.observation_space_dim
        individual_states = []
        single_obs_dim = self.config.observation_space_dim
        
        # ���ά���Ƿ���ȷ
        if len(state_obs) != self.num_drones * single_obs_dim:
            print(f"����: ״̬ά�Ȳ�ƥ��! ���� {self.num_drones * single_obs_dim}��ʵ�� {len(state_obs)}")
        
        for i in range(self.num_drones):
            start_idx = i * single_obs_dim
            end_idx = start_idx + single_obs_dim
            # ���ӷ�Χ���
            if end_idx <= len(state_obs):
                individual_states.append(state_obs[start_idx:end_idx])
            else:
                # �������������Χ��ʹ��������
                individual_states.append(torch.zeros(single_obs_dim, device=self.device))
        
        # Update high-level goals if needed
        if self.high_level_steps >= self.high_level_interval or any(g is None for g in self.high_level_goals):
            self.high_level_steps = 0
            self.high_level_goals = self.high_level_policy(state_obs, image_obs, deterministic)
        
        # Update mid-level goals if needed
        if self.mid_level_steps >= self.mid_level_interval or any(g is None for g in self.mid_level_goals):
            self.mid_level_steps = 0
            
            # Update mid-level goals for each drone
            for i in range(self.num_drones):
                # Combine individual state with high-level goal
                drone_state = individual_states[i]
                high_goal = self.high_level_goals[i]
                
                # If high-level goal is not set yet, use target position as goal
                if high_goal is None:
                    target_pos = torch.FloatTensor(self.config.target_pos).to(self.device)
                    current_pos = drone_state[:3]  # Extract position from state
                    high_goal = target_pos - current_pos
                
                # Get mid-level goal (waypoint)
                mid_input = torch.cat([drone_state, high_goal])
                self.mid_level_goals[i] = self.mid_level_policy(
                    mid_input, 
                    image_obs[i] if image_obs.dim() > 3 else image_obs,
                    deterministic
                )
        
        # Get low-level actions for each drone
        actions = []
        for i in range(self.num_drones):
            # Combine individual state with mid-level goal
            drone_state = individual_states[i]
            mid_goal = self.mid_level_goals[i]
            
            # If mid-level goal is not set yet, use neutral action
            if mid_goal is None:
                actions.append(torch.zeros(self.config.action_space_dim, device=self.device))
                continue
            
            # Get low-level action
            low_input = torch.cat([drone_state, mid_goal])
            action = self.low_level_policy(
                low_input, 
                image_obs[i] if image_obs.dim() > 3 else image_obs,
                deterministic
            )
            actions.append(action)
        
        # Combine actions for all drones
        combined_actions = torch.cat(actions)
        
        return combined_actions.cpu().numpy()
    
    def save(self, path):
        """Save model checkpoints."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'high_level_policy': self.high_level_policy.state_dict(),
            'mid_level_policy': self.mid_level_policy.state_dict(),
            'low_level_policy': self.low_level_policy.state_dict(),
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoints."""
        if not os.path.exists(path):
            print(f"Checkpoint not found at {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.high_level_policy.load_state_dict(checkpoint['high_level_policy'])
        self.mid_level_policy.load_state_dict(checkpoint['mid_level_policy'])
        self.low_level_policy.load_state_dict(checkpoint['low_level_policy'])
        
        print(f"Model loaded from {path}")
        return True
    
    def evaluate(self, env, num_episodes=10):
        """Evaluate the agent."""
        total_rewards = []
        success_rate = 0
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                # Select action
                with torch.no_grad():
                    action = self.act(obs, deterministic=True)
                
                # Take action
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                # Check if successfully reached target
                if any(d < self.config.target_radius for d in info["distances_to_target"]):
                    success_rate += 1
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(steps)
        
        # Calculate metrics
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        success_rate = success_rate / num_episodes
        mean_episode_length = np.mean(episode_lengths)
        
        metrics = {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "success_rate": success_rate,
            "mean_episode_length": mean_episode_length
        }
        
        return metrics
