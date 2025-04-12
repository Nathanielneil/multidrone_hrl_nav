import numpy as np
import torch

class RewardFunctions:
    """Collection of reward functions for multi-drone exploration."""
    
    @staticmethod
    def distance_reward(current_pos, target_pos, previous_pos=None, scale=1.0):
        """
        Reward based on distance to target.
        
        Args:
            current_pos: Current position [x, y, z]
            target_pos: Target position [x, y, z]
            previous_pos: Previous position [x, y, z] (optional)
            scale: Scaling factor
        
        Returns:
            Distance-based reward
        """
        # Calculate current distance to target
        current_distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        if previous_pos is not None:
            # Calculate previous distance to target
            previous_distance = np.linalg.norm(np.array(previous_pos) - np.array(target_pos))
            
            # Reward is the improvement in distance
            reward = scale * (previous_distance - current_distance)
        else:
            # Reward is negative distance (to encourage getting closer)
            reward = -scale * current_distance
        
        return reward
    
    @staticmethod
    def formation_reward(positions, desired_distance, scale=1.0):
        """
        Reward for maintaining formation.
        
        Args:
            positions: List of drone positions [[x1, y1, z1], [x2, y2, z2], ...]
            desired_distance: Desired distance between drones
            scale: Scaling factor
        
        Returns:
            Formation reward
        """
        num_drones = len(positions)
        
        if num_drones < 2:
            return 0.0
        
        # Calculate distances between all pairs of drones
        distances = []
        for i in range(num_drones):
            for j in range(i+1, num_drones):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                distances.append(dist)
        
        # Calculate deviation from desired distance
        deviations = [abs(d - desired_distance) for d in distances]
        mean_deviation = sum(deviations) / len(deviations)
        
        # Reward decreases as deviation increases
        reward = scale * np.exp(-mean_deviation)
        
        return reward
    
    @staticmethod
    def exploration_reward(visited_cells, current_cell, scale=1.0):
        """
        Reward for exploring new areas.
        
        Args:
            visited_cells: Set of already visited cells
            current_cell: Current cell (e.g., discretized position)
            scale: Scaling factor
        
        Returns:
            Exploration reward
        """
        if current_cell in visited_cells:
            # No reward for revisiting cells
            return 0.0
        else:
            # Reward for visiting new cells
            return scale
    
    @staticmethod
    def collision_penalty(collision_detected, scale=10.0):
        """
        Penalty for collisions.
        
        Args:
            collision_detected: Whether collision is detected
            scale: Scaling factor
        
        Returns:
            Collision penalty
        """
        return -scale if collision_detected else 0.0
    
    @staticmethod
    def success_reward(distance_to_target, success_threshold, scale=20.0):
        """
        Reward for reaching target.
        
        Args:
            distance_to_target: Distance to target
            success_threshold: Threshold for success
            scale: Scaling factor
        
        Returns:
            Success reward
        """
        return scale if distance_to_target < success_threshold else 0.0
    
    @staticmethod
    def efficiency_penalty(step, max_steps, scale=0.01):
        """
        Penalty for taking too many steps.
        
        Args:
            step: Current step
            max_steps: Maximum steps allowed
            scale: Scaling factor
        
        Returns:
            Efficiency penalty
        """
        return -scale  # Constant penalty per step
    
    @staticmethod
    def cooperative_reward(individual_rewards, scale=0.5):
        """
        Reward based on overall team performance.
        
        Args:
            individual_rewards: List of individual drone rewards
            scale: Scaling factor
        
        Returns:
            Cooperative reward
        """
        # Average of individual rewards
        avg_reward = sum(individual_rewards) / len(individual_rewards)
        
        # Additional reward if all drones are doing well
        if all(r > 0 for r in individual_rewards):
            boost = scale * avg_reward
        else:
            boost = 0.0
        
        return boost
    
    @staticmethod
    def combined_reward(
        current_positions, 
        target_pos, 
        previous_positions=None, 
        collision_detected=False,
        step=0,
        max_steps=1000,
        visited_cells=None,
        current_cells=None,
        config=None
    ):
        """
        Combined reward function.
        
        Args:
            current_positions: List of current drone positions
            target_pos: Target position
            previous_positions: List of previous drone positions (optional)
            collision_detected: Whether collision is detected
            step: Current step
            max_steps: Maximum steps allowed
            visited_cells: Set of already visited cells
            current_cells: List of current cells
            config: Configuration object
        
        Returns:
            Combined reward
        """
        if config is None:
            # Default parameters
            distance_scale = 1.0
            formation_scale = 0.5
            exploration_scale = 0.1
            collision_scale = 10.0
            success_scale = 20.0
            efficiency_scale = 0.01
            formation_distance = 5.0
            success_threshold = 5.0
        else:
            # Use parameters from config
            distance_scale = config.reward_target_factor
            formation_scale = config.reward_formation_factor
            exploration_scale = 0.1
            collision_scale = abs(config.reward_collision_penalty)
            success_scale = config.reward_success_bonus
            efficiency_scale = abs(config.reward_step_penalty)
            formation_distance = 5.0
            success_threshold = config.target_radius
        
        # Calculate individual distance rewards
        distance_rewards = []
        for i, current_pos in enumerate(current_positions):
            if previous_positions is not None:
                prev_pos = previous_positions[i]
            else:
                prev_pos = None
            
            reward = RewardFunctions.distance_reward(
                current_pos, target_pos, prev_pos, distance_scale
            )
            distance_rewards.append(reward)
        
        # Average distance reward
        avg_distance_reward = sum(distance_rewards) / len(distance_rewards)
        
        # Formation reward
        formation_reward = RewardFunctions.formation_reward(
            current_positions, formation_distance, formation_scale
        )
        
        # Exploration reward
        exploration_reward = 0.0
        if visited_cells is not None and current_cells is not None:
            for current_cell in current_cells:
                reward = RewardFunctions.exploration_reward(
                    visited_cells, current_cell, exploration_scale
                )
                exploration_reward += reward
                if reward > 0:
                    visited_cells.add(current_cell)
        
        # Collision penalty
        collision_reward = RewardFunctions.collision_penalty(
            collision_detected, collision_scale
        )
        
        # Success reward
        success_reward = 0.0
        for current_pos in current_positions:
            distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
            reward = RewardFunctions.success_reward(
                distance, success_threshold, success_scale
            )
            success_reward += reward
        
        # Efficiency penalty
        efficiency_reward = RewardFunctions.efficiency_penalty(
            step, max_steps, efficiency_scale
        )
        
        # Combine rewards
        total_reward = (
            avg_distance_reward +
            formation_reward +
            exploration_reward +
            collision_reward +
            success_reward +
            efficiency_reward
        )
        
        return total_reward

# Differentiable reward functions for end-to-end learning
class DifferentiableRewards:
    """Collection of differentiable reward functions for end-to-end learning."""
    
    @staticmethod
    def distance_reward(current_pos, target_pos, previous_pos=None, scale=1.0):
        """
        Differentiable distance reward.
        
        Args:
            current_pos: Current position tensor
            target_pos: Target position tensor
            previous_pos: Previous position tensor (optional)
            scale: Scaling factor
        
        Returns:
            Distance-based reward tensor
        """
        # Calculate current distance to target
        current_distance = torch.norm(current_pos - target_pos, dim=-1)
        
        if previous_pos is not None:
            # Calculate previous distance to target
            previous_distance = torch.norm(previous_pos - target_pos, dim=-1)
            
            # Reward is the improvement in distance
            reward = scale * (previous_distance - current_distance)
        else:
            # Reward is negative distance (to encourage getting closer)
            reward = -scale * current_distance
        
        return reward
    
    @staticmethod
    def formation_reward(positions, desired_distance, scale=1.0):
        """
        Differentiable formation reward.
        
        Args:
            positions: Tensor of drone positions [num_drones, 3]
            desired_distance: Desired distance between drones
            scale: Scaling factor
        
        Returns:
            Formation reward tensor
        """
        num_drones = positions.shape[0]
        
        if num_drones < 2:
            return torch.tensor(0.0, device=positions.device)
        
        # Calculate distances between all pairs of drones
        distances = []
        for i in range(num_drones):
            for j in range(i+1, num_drones):
                dist = torch.norm(positions[i] - positions[j])
                distances.append(dist)
        
        # Stack distances
        distances = torch.stack(distances)
        
        # Calculate deviation from desired distance
        deviations = torch.abs(distances - desired_distance)
        mean_deviation = torch.mean(deviations)
        
        # Reward decreases as deviation increases
        reward = scale * torch.exp(-mean_deviation)
        
        return reward
    
    @staticmethod
    def collision_penalty(collision_probabilities, scale=10.0):
        """
        Differentiable collision penalty.
        
        Args:
            collision_probabilities: Tensor of collision probabilities
            scale: Scaling factor
        
        Returns:
            Collision penalty tensor
        """
        return -scale * collision_probabilities
    
    @staticmethod
    def success_reward(distances_to_target, success_threshold, scale=20.0):
        """
        Differentiable success reward.
        
        Args:
            distances_to_target: Tensor of distances to target
            success_threshold: Threshold for success
            scale: Scaling factor
        
        Returns:
            Success reward tensor
        """
        # Sigmoid function to smooth the transition
        success_probs = torch.sigmoid(-(distances_to_target - success_threshold) * 10)
        return scale * success_probs
