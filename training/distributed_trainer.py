import os
import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
from tqdm import tqdm
from .ppo_trainer import PPOTrainer
import torch.optim as optim
from collections import deque

class DistributedTrainer:
    """Distributed trainer for multi-drone hierarchical RL."""
    
    def __init__(self, config, envs, agent, device):
        """
        Initialize the distributed trainer.
        
        Args:
            config: Configuration object
            envs: List of environments (one per process)
            agent: Hierarchical agent
            device: Torch device (CPU or GPU)
        """
        self.config = config
        self.envs = envs
        self.agent = agent
        self.device = device
        
        # Number of environments
        self.num_envs = len(envs)
        
        # Create PPO trainers
        self.trainers = [
            PPOTrainer(
                config,
                agent,
                device,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                clip_range=config.clip_range,
                ent_coef=config.ent_coef,
                value_coef=config.value_coef,
                max_grad_norm=config.max_grad_norm,
                batch_size=config.batch_size,
                n_epochs=config.n_epochs
            )
            for _ in range(self.num_envs)
        ]
        
        # Metrics for logging
        self.metrics = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'target_distances': deque(maxlen=100),
            'success_rate': deque(maxlen=100)
        }
        
        # Create optimizers
        self.high_level_optimizer = optim.Adam(
            self.agent.high_level_policy.parameters(),
            lr=config.learning_rate
        )
        self.mid_level_optimizer = optim.Adam(
            self.agent.mid_level_policy.parameters(),
            lr=config.learning_rate
        )
        self.low_level_optimizer = optim.Adam(
            self.agent.low_level_policy.parameters(),
            lr=config.learning_rate
        )
    
    def train(self):
        """Train the agent using distributed training."""
        print(f"Starting training with {self.num_envs} environments")
        
        # Total timesteps
        total_timesteps = self.config.total_timesteps
        timesteps_per_env = total_timesteps // self.num_envs
        
        # Initialize environments
        observations = [env.reset() for env in self.envs]
        
        # Training loop
        global_step = 0
        start_time = time.time()
        
        for step in tqdm(range(timesteps_per_env), desc="Training"):
            # Collect experiences from all environments
            actions = []
            for i, obs in enumerate(observations):
                action = self.agent.act(obs)
                actions.append(action)
            
            # Step environments
            next_observations = []
            rewards = []
            dones = []
            infos = []
            
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                next_obs, reward, done, info = env.step(action)
                next_observations.append(next_obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                
                # Update trainer with experience
                self.trainers[i].add_experience(
                    observations[i], action, reward, next_obs, done, info
                )
                
                # Reset environment if episode is done
                if done:
                    next_observations[i] = env.reset()
                    
                    # Log metrics
                    self.metrics['episode_rewards'].append(info.get('episode_reward', 0))
                    self.metrics['episode_lengths'].append(info.get('step', 0))
                    
                    # Calculate average distance to target
                    if 'distances_to_target' in info:
                        avg_distance = np.mean(info['distances_to_target'])
                        self.metrics['target_distances'].append(avg_distance)
                    
                    # Calculate success rate
                    success = any(d < self.config.target_radius for d in info.get('distances_to_target', [np.inf]))
                    self.metrics['success_rate'].append(float(success))
            
            # Update observations
            observations = next_observations
            
            # Train after collecting enough experience
            if step > 0 and step % self.config.batch_size == 0:
                # Parallel training across environments
                gradients = []
                for trainer in self.trainers:
                    # Get gradients from each trainer
                    high_grad, mid_grad, low_grad = trainer.compute_gradients()
                    gradients.append((high_grad, mid_grad, low_grad))
                
                # Average gradients
                avg_high_grad = self._average_gradients([g[0] for g in gradients])
                avg_mid_grad = self._average_gradients([g[1] for g in gradients])
                avg_low_grad = self._average_gradients([g[2] for g in gradients])
                
                # Apply gradients
                self._apply_gradients(
                    self.high_level_optimizer, 
                    self.agent.high_level_policy, 
                    avg_high_grad
                )
                self._apply_gradients(
                    self.mid_level_optimizer,
                    self.agent.mid_level_policy,
                    avg_mid_grad
                )
                self._apply_gradients(
                    self.low_level_optimizer,
                    self.agent.low_level_policy,
                    avg_low_grad
                )
            
            # Update global step
            global_step += self.num_envs
            
            # Logging
            if step > 0 and step % self.config.log_freq == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = global_step / elapsed_time
                
                # Calculate average metrics
                avg_reward = np.mean(self.metrics['episode_rewards']) if self.metrics['episode_rewards'] else 0
                avg_length = np.mean(self.metrics['episode_lengths']) if self.metrics['episode_lengths'] else 0
                avg_distance = np.mean(self.metrics['target_distances']) if self.metrics['target_distances'] else float('inf')
                success_rate = np.mean(self.metrics['success_rate']) if self.metrics['success_rate'] else 0
                
                print(f"Step: {global_step}, FPS: {steps_per_sec:.2f}")
                print(f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
                print(f"Avg Distance to Target: {avg_distance:.2f}, Success Rate: {success_rate:.2%}")
            
            # Save checkpoint
            if step > 0 and step % self.config.save_freq == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_{global_step}.pt"
                )
                self.agent.save(checkpoint_path)
            
            # Evaluate
            if step > 0 and step % self.config.eval_freq == 0:
                eval_metrics = self.agent.evaluate(self.envs[0], num_episodes=5)
                print("Evaluation:")
                for key, value in eval_metrics.items():
                    print(f"{key}: {value}")
        
        # Save final model
        final_checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            "checkpoint_final.pt"
        )
        self.agent.save(final_checkpoint_path)
        
        print("Training completed!")
    
    def _average_gradients(self, gradients):
        """Average gradients from multiple trainers."""
        if not gradients:
            return None
        
        avg_grads = []
        for grad_tuple in zip(*gradients):
            # Filter out None gradients
            valid_grads = [g for g in grad_tuple if g is not None]
            if not valid_grads:
                avg_grads.append(None)
                continue
            
            # Stack and average
            stacked = torch.stack(valid_grads)
            avg = torch.mean(stacked, dim=0)
            avg_grads.append(avg)
        
        return avg_grads
    
    def _apply_gradients(self, optimizer, model, gradients):
        """Apply gradients to a model."""
        if not gradients:
            return
        
        optimizer.zero_grad()
        
        # Manually set gradients
        for param, grad in zip(model.parameters(), gradients):
            if grad is not None:
                param.grad = grad
        
        # Apply gradient clipping
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
