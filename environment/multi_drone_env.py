import numpy as np
import airsim
import time
import gym
from gym import spaces
from .airsim_env import AirSimDroneEnv

class MultiDroneEnv(gym.Env):
    """Environment for controlling multiple drones in AirSim."""
    
    def __init__(self, config, env_id=0):
        """
        Initialize the multi-drone environment.
        
        Args:
            config: Configuration object
            env_id: Environment ID for parallel environments
        """
        self.config = config
        self.env_id = env_id
        self.drone_names = config.drone_names
        self.num_drones = len(self.drone_names)
        self.target_pos = config.target_pos
        self.max_steps_per_episode = config.max_steps_per_episode
        self.current_step = 0
        
        # Create drone environments
        ip_address = f"127.0.0.{1 + env_id}"  # Different IP for each environment
        self.drones = [
            AirSimDroneEnv(drone_name, self.target_pos, ip_address)
            for drone_name in self.drone_names
        ]
        
        # Define observation and action spaces
        # Each drone has its own state space and action space
        self.single_drone_obs_dim = self.drones[0].observation_space.shape[0]
        
        # Total observation includes all drones' states plus their relative positions to each other
        self.relative_pos_dim = 3 * (self.num_drones - 1) * self.num_drones // 2
        total_obs_dim = self.single_drone_obs_dim * self.num_drones + self.relative_pos_dim
        
        self.observation_space = spaces.Dict({
            "states": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(total_obs_dim,), 
                dtype=np.float32
            ),
            "images": spaces.Box(
                low=0, high=1,
                shape=(self.num_drones, config.image_height, config.image_width, config.image_channels),
                dtype=np.float32
            )
        })
        
        # Action space for all drones
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_drones * 4,),  # 4 actions per drone
            dtype=np.float32
        )
        
        # Formation parameters
        self.formation_distance = 5.0  # Desired distance between drones
        
        # Initialize
        self.reset()
    
    def reset(self):
        """Reset the environment."""
        # Reset all drones
        observations = [drone.reset() for drone in self.drones]
        
        # Reset step counter
        self.current_step = 0
        
        # Combine observations
        return self._combine_observations(observations)
    
    def step(self, actions):
        """
        Take a step in the environment.
        
        Args:
            actions: Combined actions for all drones [drone1_actions, drone2_actions, ...]
        
        Returns:
            observation, reward, done, info
        """
        self.current_step += 1
        
        # Split actions for each drone
        individual_actions = np.split(actions, self.num_drones)
        
        # Execute actions and collect results
        results = []
        for i, drone in enumerate(self.drones):
            obs, rew, done, info = drone.step(individual_actions[i])
            results.append((obs, rew, done, info))
        
        # Combine observations, rewards, dones
        observations = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        
        # Calculate formation reward
        formation_reward = self._calculate_formation_reward()
        
        # Combine rewards: individual rewards + formation reward
        total_reward = sum(rewards) + self.config.reward_formation_factor * formation_reward
        
        # Check if episode is done
        done = any(dones) or self.current_step >= self.max_steps_per_episode
        
        # Combine all information
        combined_obs = self._combine_observations(observations)
        combined_info = {
            "individual_rewards": rewards,
            "formation_reward": formation_reward,
            "individual_dones": dones,
            "step": self.current_step,
            "drone_positions": [info["position"] for info in infos],
            "distances_to_target": [info["distance_to_target"] for info in infos]
        }
        
        return combined_obs, total_reward, done, combined_info
    
    def _combine_observations(self, observations):
        """Combine observations from all drones."""
        # Extract state vectors
        state_vectors = [obs["state"] for obs in observations]
        
        # Extract images
        images = np.array([obs["image"] for obs in observations])
        
        # Calculate relative positions between drones
        relative_positions = []
        for i in range(self.num_drones):
            for j in range(i+1, self.num_drones):
                # Extract position from state vectors (first 3 elements)
                pos_i = state_vectors[i][:3]
                pos_j = state_vectors[j][:3]
                # Calculate relative position
                rel_pos = pos_j - pos_i
                relative_positions.append(rel_pos)
        
        # Flatten relative positions
        flat_relative_positions = np.concatenate(relative_positions)
        
        # Combine all state information
        combined_states = np.concatenate(state_vectors + [flat_relative_positions])
        
        return {
            "states": combined_states.astype(np.float32),
            "images": images
        }
    
    def _calculate_formation_reward(self):
        """Calculate reward for maintaining formation."""
        # Get positions of all drones
        positions = []
        for drone in self.drones:
            drone_state = drone.client.getMultirotorState(vehicle_name=drone.drone_name)
            pos = drone_state.kinematics_estimated.position
            positions.append(np.array([pos.x_val, pos.y_val, pos.z_val]))
        
        # Calculate distances between drones
        distances = []
        for i in range(self.num_drones):
            for j in range(i+1, self.num_drones):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        # Calculate reward based on how close the distances are to the desired formation distance
        formation_errors = [abs(d - self.formation_distance) for d in distances]
        mean_error = sum(formation_errors) / len(formation_errors)
        
        # Reward decreases as error increases (negative reward for error)
        formation_reward = -mean_error
        
        return formation_reward
    
    def close(self):
        """Close the environment."""
        for drone in self.drones:
            drone.close()
