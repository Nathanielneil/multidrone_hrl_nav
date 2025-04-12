import numpy as np
import airsim
import time
import cv2
from gym import spaces

class AirSimDroneEnv:
    """Base AirSim environment for a single drone"""
    
    def __init__(self, drone_name, target_pos=None, ip_address="127.0.0.1"):
        # Initialize AirSim client
        self.drone_name = drone_name
        self.client = airsim.MultirotorClient(ip_address)
        self.target_pos = target_pos or [50.0, 50.0, -10.0]  # Default target if none provided
        
        # Connect to AirSim
        self.client.confirmConnection()
        
        # Enable API control and arm the drone
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        
        # Initialize observation and action spaces
        self.observation_space = spaces.Box(
            low=np.array([-100, -100, -100, -10, -10, -10, -1, -1, -1, -1, -100, -100, -100]),
            high=np.array([100, 100, 100, 10, 10, 10, 1, 1, 1, 1, 100, 100, 100]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Maximum velocity limits
        self.vx_max = 5.0
        self.vy_max = 5.0
        self.vz_max = 2.0
        self.yaw_rate_max = 1.0
        
        # Collision detection
        self.collision_detected = False
        
        # Image observation
        self.image_width = 84
        self.image_height = 84
        
        # Initialize the environment
        self.reset()
    
    def reset(self):
        """Reset the environment."""
        # Reset drone to initial position
        self.client.reset()
        self.client.enableApiControl(True, self.drone_name)
        self.client.armDisarm(True, self.drone_name)
        
        # Take off and hover
        self.client.takeoffAsync(vehicle_name=self.drone_name).join()
        
        # Reset collision flag
        self.collision_detected = False
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment."""
        # Normalize actions from [-1, 1] to actual velocity commands
        vx = action[0] * self.vx_max
        vy = action[1] * self.vy_max
        vz = action[2] * self.vz_max
        yaw_rate = action[3] * self.yaw_rate_max
        
        # Send velocity command to drone
        self.client.moveByVelocityAsync(
            vx, vy, vz, 
            duration=0.5,  # Duration for this command
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=self.drone_name
        )
        
        # Small sleep to allow physics to update
        time.sleep(0.05)
        
        # Get observation, reward, done
        observation = self._get_observation()
        reward, done = self._compute_reward()
        
        # Check for collision
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_name)
        if collision_info.has_collided:
            self.collision_detected = True
            done = True
            reward -= 10.0  # Collision penalty
        
        info = {
            "collision": self.collision_detected,
            "position": self.client.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated.position,
            "distance_to_target": self._get_distance_to_target()
        }
        
        return observation, reward, done, info
    
    def _get_observation(self):
        """Get the current observation."""
        # Get drone state
        drone_state = self.client.getMultirotorState(vehicle_name=self.drone_name)
        
        # Extract position
        pos = drone_state.kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        
        # Extract velocity
        vel = drone_state.kinematics_estimated.linear_velocity
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        # Extract orientation (quaternion)
        orientation = drone_state.kinematics_estimated.orientation
        q = np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val])
        
        # Calculate relative position to target
        target = np.array(self.target_pos)
        relative_pos = target - position
        
        # Get image from camera
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.drone_name)
        
        image = self._process_image(responses[0])
        
        # Flatten everything into a single observation vector
        # (We'll handle the image separately in the hierarchical agent)
        obs = np.concatenate([position, velocity, q, relative_pos])
        
        return {
            "state": obs.astype(np.float32),
            "image": image
        }
    
    def _process_image(self, response):
        """Process image from AirSim."""
        # Convert image from AirSim to numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        
        # Resize image to desired dimensions
        img_resized = cv2.resize(img_rgb, (self.image_width, self.image_height))
        
        # Normalize image
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    
    def _get_distance_to_target(self):
        """Calculate distance to target."""
        pos = self.client.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        target = np.array(self.target_pos)
        
        return np.linalg.norm(target - position)
    
    def _compute_reward(self):
        """Compute reward and check if episode is done."""
        # Get current distance to target
        distance = self._get_distance_to_target()
        
        # Check if reached target (within 5 meters)
        reached_target = distance < 5.0
        
        # Reward is negative distance to target (to encourage getting closer)
        reward = -0.01 * distance
        
        # Bonus for reaching target
        if reached_target:
            reward += 20.0
        
        # Episode is done if reached target or collision
        done = reached_target or self.collision_detected
        
        return reward, done
    
    def get_camera_image(self):
        """Get RGB image from front camera."""
        responses = self.client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ], vehicle_name=self.drone_name)
        
        return self._process_image(responses[0])
    
    def get_imu_data(self):
        """Get IMU data."""
        imu_data = self.client.getImuData(vehicle_name=self.drone_name)
        
        # Extract angular velocity
        angular_velocity = np.array([
            imu_data.angular_velocity.x_val,
            imu_data.angular_velocity.y_val,
            imu_data.angular_velocity.z_val
        ])
        
        # Extract linear acceleration
        linear_acceleration = np.array([
            imu_data.linear_acceleration.x_val,
            imu_data.linear_acceleration.y_val,
            imu_data.linear_acceleration.z_val
        ])
        
        return {
            "angular_velocity": angular_velocity,
            "linear_acceleration": linear_acceleration
        }
    
    def close(self):
        """Close the environment."""
        self.client.armDisarm(False, self.drone_name)
        self.client.enableApiControl(False, self.drone_name)