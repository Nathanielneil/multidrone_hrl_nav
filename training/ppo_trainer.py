import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

class PPOTrainer:
    """Proximal Policy Optimization (PPO) trainer for hierarchical RL."""
    
    def __init__(
        self,
        config,
        agent,
        device,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        batch_size=64,
        n_epochs=10
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            config: Configuration object
            agent: Hierarchical agent
            device: Torch device (CPU or GPU)
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_range: PPO clipping parameter
            ent_coef: Entropy coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm
            batch_size: Batch size for training
            n_epochs: Number of epochs for training
        """
        self.config = config
        self.agent = agent
        self.device = device
        
        # PPO parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Experience buffer
        self.buffer = {
            'states': [],
            'images': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Validation
        self.n_steps = 0
    
    def add_experience(self, obs, action, reward, next_obs, done, info):
        """
        Add experience to the buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether the episode is done
            info: Additional information
        """
        # Get state and image from observation
        state = torch.FloatTensor(obs['states']).to(self.device)
        if 'images' in obs:
            image = torch.FloatTensor(obs['images']).to(self.device)
        else:
            image = None
        
        # Convert action to tensor
        action_tensor = torch.FloatTensor(action).to(self.device)
        
        # Get log probability and value for current state-action
        with torch.no_grad():
            # High-level policy evaluation
            high_values, high_log_probs, _ = self.agent.high_level_policy.evaluate_actions(
                state, image, action_tensor[:self.agent.num_drones * 3]
            )
            
            # Mid-level policy evaluation (approximate, would need to be more precise in full implementation)
            mid_values, mid_log_probs, _ = self.agent.mid_level_policy.evaluate_actions(
                state[:self.config.observation_space_dim], image, action_tensor[self.agent.num_drones * 3:self.agent.num_drones * 6]
            )
            
            # Low-level policy evaluation
            low_values, low_log_probs, _ = self.agent.low_level_policy.evaluate_actions(
                state[:self.config.observation_space_dim], image, action_tensor[-self.config.action_space_dim:]
            )
            
            # Combine values and log probs
            values = (high_values + mid_values + low_values) / 3
            log_probs = (high_log_probs + mid_log_probs + low_log_probs) / 3
        
        # Add to buffer
        self.buffer['states'].append(state)
        if image is not None:
            self.buffer['images'].append(image)
        else:
            self.buffer['images'].append(None)
        self.buffer['actions'].append(action_tensor)
        self.buffer['rewards'].append(torch.FloatTensor([reward]).to(self.device))
        self.buffer['values'].append(values)
        self.buffer['log_probs'].append(log_probs)
        self.buffer['dones'].append(torch.FloatTensor([float(done)]).to(self.device))
        
        self.n_steps += 1
    
    def compute_returns_and_advantages(self):
        """Compute returns and advantages using Generalized Advantage Estimation (GAE)."""
        # Get final value estimate
        with torch.no_grad():
            last_state = self.buffer['states'][-1]
            last_image = self.buffer['images'][-1]
            
            # High-level policy
            high_next_value = self.agent.high_level_policy.value_net(
                torch.cat([
                    self.agent.high_level_policy.state_encoder(last_state),
                    torch.zeros(self.agent.high_level_policy.cnn.fc.out_features, device=self.device)
                ])
            )
            
            # Mid-level policy
            mid_next_value = self.agent.mid_level_policy.value_net(last_state[:self.config.observation_space_dim])
            
            # Low-level policy
            low_next_value = self.agent.low_level_policy.value_net(last_state[:self.config.observation_space_dim])
            
            # Combined value
            next_value = (high_next_value + mid_next_value + low_next_value) / 3
        
        # Initialize returns and advantages
        returns = []
        advantages = []
        
        # Initialize gae
        gae = 0
        
        # Reverse iterate to compute advantage
        for t in reversed(range(len(self.buffer['rewards']))):
            # If t is the last step, use next_value, otherwise use next step's value
            if t == len(self.buffer['rewards']) - 1:
                next_value_t = next_value
                next_non_terminal = 1.0 - self.buffer['dones'][t]
            else:
                next_value_t = self.buffer['values'][t + 1]
                next_non_terminal = 1.0 - self.buffer['dones'][t]
            
            # Compute delta (TD error)
            delta = (
                self.buffer['rewards'][t] + 
                self.gamma * next_value_t * next_non_terminal - 
                self.buffer['values'][t]
            )
            
            # Update gae
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            
            # Insert at the beginning (reverse order)
            advantages.insert(0, gae)
            returns.insert(0, gae + self.buffer['values'][t])
        
        return torch.cat(returns), torch.cat(advantages)
    
    def compute_gradients(self):
        """
        Compute gradients for updating the policy.
        
        Returns:
            Gradients for the high, mid, and low-level policies
        """
        # Check if there's enough data
        if self.n_steps < self.batch_size:
            return None, None, None
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare batch data
        batch_states = torch.stack(self.buffer['states'])
        batch_actions = torch.stack(self.buffer['actions'])
        batch_log_probs = torch.cat(self.buffer['log_probs'])
        
        # Handle images (may be None)
        if all(img is not None for img in self.buffer['images']):
            batch_images = torch.stack(self.buffer['images'])
        else:
            batch_images = None
        
        # Initialize losses for each level
        high_policy_loss = 0
        high_value_loss = 0
        high_entropy_loss = 0
        
        mid_policy_loss = 0
        mid_value_loss = 0
        mid_entropy_loss = 0
        
        low_policy_loss = 0
        low_value_loss = 0
        low_entropy_loss = 0
        
        # Zero gradients
        for param in self.agent.high_level_policy.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        
        for param in self.agent.mid_level_policy.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        
        for param in self.agent.low_level_policy.parameters():
            if param.grad is not None:
                param.grad.data.zero_()
        
        # Train for multiple epochs
        for epoch in range(self.n_epochs):
            # Generate random indices
            indices = torch.randperm(self.n_steps)
            
            # Train in mini-batches
            for start in range(0, self.n_steps, self.batch_size):
                end = start + self.batch_size
                if end > self.n_steps:
                    break
                
                # Get mini-batch indices
                batch_idx = indices[start:end]
                
                # Get mini-batch data
                mb_states = batch_states[batch_idx]
                mb_actions = batch_actions[batch_idx]
                mb_returns = returns[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_log_probs = batch_log_probs[batch_idx]
                
                if batch_images is not None:
                    mb_images = batch_images[batch_idx]
                else:
                    mb_images = None
                
                # Split actions for each level
                high_actions = mb_actions[:, :self.agent.num_drones * 3]
                mid_actions = mb_actions[:, self.agent.num_drones * 3:self.agent.num_drones * 6]
                low_actions = mb_actions[:, -self.config.action_space_dim:]
                
                # Forward pass for each level
                # High-level policy
                high_values, high_log_probs, high_entropy = self.agent.high_level_policy.evaluate_actions(
                    mb_states[0], mb_images[0] if mb_images is not None else None, high_actions
                )
                
                # Mid-level policy
                mid_values, mid_log_probs, mid_entropy = self.agent.mid_level_policy.evaluate_actions(
                    mb_states[0, :self.config.observation_space_dim], 
                    mb_images[0] if mb_images is not None else None, 
                    mid_actions
                )
                
                # Low-level policy
                low_values, low_log_probs, low_entropy = self.agent.low_level_policy.evaluate_actions(
                    mb_states[0, :self.config.observation_space_dim], 
                    mb_images[0] if mb_images is not None else None, 
                    low_actions
                )
                
                # Compute ratio (importance sampling)
                high_ratio = torch.exp(high_log_probs - mb_log_probs)
                mid_ratio = torch.exp(mid_log_probs - mb_log_probs)
                low_ratio = torch.exp(low_log_probs - mb_log_probs)
                
                # Clipped surrogate loss
                high_policy_loss_1 = -mb_advantages * high_ratio
                high_policy_loss_2 = -mb_advantages * torch.clamp(high_ratio, 1 - self.clip_range, 1 + self.clip_range)
                high_policy_loss += torch.max(high_policy_loss_1, high_policy_loss_2).mean()
                
                mid_policy_loss_1 = -mb_advantages * mid_ratio
                mid_policy_loss_2 = -mb_advantages * torch.clamp(mid_ratio, 1 - self.clip_range, 1 + self.clip_range)
                mid_policy_loss += torch.max(mid_policy_loss_1, mid_policy_loss_2).mean()
                
                low_policy_loss_1 = -mb_advantages * low_ratio
                low_policy_loss_2 = -mb_advantages * torch.clamp(low_ratio, 1 - self.clip_range, 1 + self.clip_range)
                low_policy_loss += torch.max(low_policy_loss_1, low_policy_loss_2).mean()
                
                # Value loss
                high_value_loss += 0.5 * F.mse_loss(high_values, mb_returns)
                mid_value_loss += 0.5 * F.mse_loss(mid_values, mb_returns)
                low_value_loss += 0.5 * F.mse_loss(low_values, mb_returns)
                
                # Entropy loss
                high_entropy_loss -= high_entropy
                mid_entropy_loss -= mid_entropy
                low_entropy_loss -= low_entropy
        
        # Average losses over epochs and mini-batches
        n_minibatches = (self.n_steps + self.batch_size - 1) // self.batch_size
        high_policy_loss /= (self.n_epochs * n_minibatches)
        high_value_loss /= (self.n_epochs * n_minibatches)
        high_entropy_loss /= (self.n_epochs * n_minibatches)
        
        mid_policy_loss /= (self.n_epochs * n_minibatches)
        mid_value_loss /= (self.n_epochs * n_minibatches)
        mid_entropy_loss /= (self.n_epochs * n_minibatches)
        
        low_policy_loss /= (self.n_epochs * n_minibatches)
        low_value_loss /= (self.n_epochs * n_minibatches)
        low_entropy_loss /= (self.n_epochs * n_minibatches)
        
        # Total loss for each level
        high_loss = high_policy_loss + self.value_coef * high_value_loss + self.ent_coef * high_entropy_loss
        mid_loss = mid_policy_loss + self.value_coef * mid_value_loss + self.ent_coef * mid_entropy_loss
        low_loss = low_policy_loss + self.value_coef * low_value_loss + self.ent_coef * low_entropy_loss
        
        # Compute gradients
        high_loss.backward()
        mid_loss.backward()
        low_loss.backward()
        
        # Get gradients
        high_gradients = [param.grad.clone() for param in self.agent.high_level_policy.parameters() if param.grad is not None]
        mid_gradients = [param.grad.clone() for param in self.agent.mid_level_policy.parameters() if param.grad is not None]
        low_gradients = [param.grad.clone() for param in self.agent.low_level_policy.parameters() if param.grad is not None]
        
        # Clear buffer
        for key in self.buffer:
            self.buffer[key] = []
        self.n_steps = 0
        
        return high_gradients, mid_gradients, low_gradients
