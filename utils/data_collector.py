import os
import numpy as np
import torch
import time
from collections import deque

class RolloutBuffer:
    """Buffer for storing rollout data for training."""
    
    def __init__(self, buffer_size, state_dim, action_dim, num_drones, image_shape=None):
        """
        Initialize the rollout buffer.
        
        Args:
            buffer_size: Maximum size of the buffer
            state_dim: Dimension of the state
            action_dim: Dimension of the action
            num_drones: Number of drones
            image_shape: Shape of image observation (C, H, W)
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_drones = num_drones
        self.image_shape = image_shape
        self.use_images = image_shape is not None
        
        # Initialize buffers
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        
        if self.use_images:
            channels, height, width = image_shape
            self.images = np.zeros(
                (buffer_size, num_drones, channels, height, width), 
                dtype=np.float32
            )
            self.next_images = np.zeros(
                (buffer_size, num_drones, channels, height, width), 
                dtype=np.float32
            )
        
        self.current_size = 0
        self.position = 0
    
    def add(self, state, action, reward, next_state, done, image=None, next_image=None):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            image: Image observation (optional)
            next_image: Next image observation (optional)
        """
        # Add state transition
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Add image observations if available
        if self.use_images and image is not None and next_image is not None:
            self.images[self.position] = image
            self.next_images[self.position] = next_image
        
        # Update position and size
        self.position = (self.position + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Size of the batch to sample
        
        Returns:
            A batch of transitions
        """
        # Sample indices
        indices = np.random.randint(0, self.current_size, size=batch_size)
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        
        if self.use_images:
            batch['images'] = self.images[indices]
            batch['next_images'] = self.next_images[indices]
        
        return batch
    
    def clear(self):
        """Clear the buffer."""
        self.current_size = 0
        self.position = 0


class DataCollector:
    """Collector for gathering experience data."""
    
    def __init__(self, config, env, agent):
        """
        Initialize the data collector.
        
        Args:
            config: Configuration object
            env: Environment to collect data from
            agent: Agent to collect data with
        """
        self.config = config
        self.env = env
        self.agent = agent
        
        # Initialize rollout buffer
        state_dim = env.observation_space['states'].shape[0]
        action_dim = env.action_space.shape[0]
        
        if config.use_images:
            image_shape = (
                config.image_channels,
                config.image_height,
                config.image_width
            )
        else:
            image_shape = None
        
        self.buffer = RolloutBuffer(
            buffer_size=config.batch_size * 10,  # Buffer size is 10x batch size
            state_dim=state_dim,
            action_dim=action_dim,
            num_drones=config.num_drones,
            image_shape=image_shape if config.use_images else None
        )
        
        # Metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_rate = deque(maxlen=100)
    
    def collect_episode(self, deterministic=False):
        """
        Collect data for one episode.
        
        Args:
            deterministic: Whether to use deterministic action selection
        
        Returns:
            episode_reward: Total reward for the episode
            episode_length: Length of the episode
            success: Whether the episode was successful
        """
        # Reset environment
        obs = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Select action
            action = self.agent.act(obs, deterministic=deterministic)
            
            # Take action
            next_obs, reward, done, info = self.env.step(action)
            
            # Add to buffer
            if self.config.use_images:
                self.buffer.add(
                    state=obs['states'],
                    action=action,
                    reward=reward,
                    next_state=next_obs['states'],
                    done=done,
                    image=obs['images'],
                    next_image=next_obs['images']
                )
            else:
                self.buffer.add(
                    state=obs['states'],
                    action=action,
                    reward=reward,
                    next_state=next_obs['states'],
                    done=done
                )
            
            # Update observation
            obs = next_obs
            
            # Update episode reward and length
            episode_reward += reward
            episode_length += 1
        
        # Check if episode was successful
        success = any(d < self.config.target_radius for d in info['distances_to_target'])
        
        # Update metrics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.success_rate.append(float(success))
        
        return episode_reward, episode_length, success
    
    def collect_data(self, num_steps, deterministic=False):
        """
        Collect data for a specified number of steps.
        
        Args:
            num_steps: Number of steps to collect
            deterministic: Whether to use deterministic action selection
        
        Returns:
            buffer: Collected data buffer
            metrics: Collection metrics
        """
        # Reset environment
        obs = self.env.reset()
        
        # Clear buffer
        self.buffer.clear()
        
        # Collect data
        steps_collected = 0
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        episodes_completed = 0
        
        current_episode_reward = 0
        current_episode_length = 0
        
        while steps_collected < num_steps:
            # Select action
            action = self.agent.act(obs, deterministic=deterministic)
            
            # Take action
            next_obs, reward, done, info = self.env.step(action)
            
            # Add to buffer
            if self.config.use_images:
                self.buffer.add(
                    state=obs['states'],
                    action=action,
                    reward=reward,
                    next_state=next_obs['states'],
                    done=done,
                    image=obs['images'],
                    next_image=next_obs['images']
                )
            else:
                self.buffer.add(
                    state=obs['states'],
                    action=action,
                    reward=reward,
                    next_state=next_obs['states'],
                    done=done
                )
            
            # Update observation
            obs = next_obs
            
            # Update episode reward and length
            current_episode_reward += reward
            current_episode_length += 1
            
            # Increment steps collected
            steps_collected += 1
            
            # Handle episode termination
            if done:
                # Check if episode was successful
                success = any(d < self.config.target_radius for d in info['distances_to_target'])
                
                # Update episode metrics
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                if success:
                    success_count += 1
                
                # Reset episode metrics
                current_episode_reward = 0
                current_episode_length = 0
                
                # Increment episode counter
                episodes_completed += 1
                
                # Reset environment
                obs = self.env.reset()
        
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0,
            'success_rate': success_count / episodes_completed if episodes_completed > 0 else 0,
            'episodes_completed': episodes_completed,
            'steps_collected': steps_collected
        }
        
        return self.buffer, metrics
    
    def get_metrics(self):
        """Get current metrics."""
        return {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'success_rate': np.mean(self.success_rate) if self.success_rate else 0
        }
