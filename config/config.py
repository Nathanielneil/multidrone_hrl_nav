import os
import json

class Config:
    def __init__(self, config_name='default'):
        # Basic settings
        self.config_name = config_name
        
        # Environment settings
        self.num_drones = 3
        self.num_envs = 4  # Number of parallel environments
        self.drone_names = [f"Drone{i+1}" for i in range(self.num_drones)]
        self.max_steps_per_episode = 1000
        self.target_pos = [50.0, 50.0, -10.0]  # Target position [x, y, z] in AirSim coordinates
        self.target_radius = 5.0  # Success radius around target
        self.observation_space_dim = 13  # Position (3), Velocity (3), Attitude (3), Target relative pos (3)
        self.action_space_dim = 4  # Velocity commands (vx, vy, vz, yaw_rate)
        
        # Image settings
        self.use_images = True
        self.image_width = 84
        self.image_height = 84
        self.image_channels = 3
        
        # Hierarchical RL settings
        self.high_level_action_interval = 10  # High-level policy updates every N steps
        self.mid_level_action_interval = 5    # Mid-level policy updates every N steps
        
        # Training settings
        self.total_timesteps = 1_000_000
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        self.batch_size = 64
        self.n_epochs = 10
        
        # Network architecture
        self.high_level_hidden_dim = 256
        self.mid_level_hidden_dim = 128
        self.low_level_hidden_dim = 64
        
        # CNN architecture for image processing
        self.cnn_features_dim = 512
        
        # Saving and logging
        self.save_freq = 10000
        self.log_freq = 1000
        self.eval_freq = 10000
        self.checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        
        # Reward settings
        self.reward_target_factor = 1.0      # Reward factor for getting closer to target
        self.reward_collision_penalty = -10.0  # Penalty for collision
        self.reward_success_bonus = 20.0     # Bonus for reaching target
        self.reward_step_penalty = -0.01     # Small penalty for each step to encourage efficiency
        self.reward_formation_factor = 0.5   # Reward factor for maintaining formation
        
        # Load config from file if it exists
        self.load_config()
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def load_config(self):
        """Load configuration from file if it exists"""
        config_path = os.path.join(os.getcwd(), 'config', f'{self.config_name}.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
                for key, value in config_dict.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
    
    def save_config(self):
        """Save configuration to file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        config_path = os.path.join(os.getcwd(), 'config', f'{self.config_name}.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)