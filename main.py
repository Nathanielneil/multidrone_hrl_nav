import argparse
import os
import torch

from config.config import Config
from environment.multi_drone_env import MultiDroneEnv
from models.hierarchical_agent import HierarchicalAgent
from training.distributed_trainer import DistributedTrainer
from evaluation.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Drone Hierarchical RL')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'visualize'],
                        help='Mode: train, eval, or visualize')
    parser.add_argument('--config', type=str, default='default', help='Configuration name')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for evaluation')
    parser.add_argument('--num_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--target_pos', nargs=3, type=float, default=[50.0, 50.0, -10.0],
                        help='Target position [x, y, z] in AirSim coordinates')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    config.num_envs = args.num_envs
    config.target_pos = args.target_pos
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environments
    envs = [MultiDroneEnv(config, i) for i in range(config.num_envs)]
    
    # Create agent
    agent = HierarchicalAgent(config, device)
    
    if args.mode == 'train':
        # Create trainer
        trainer = DistributedTrainer(config, envs, agent, device)
        
        # Train the agent
        trainer.train()
        
    elif args.mode == 'eval':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for evaluation")
        
        # Load checkpoint
        agent.load(args.checkpoint)
        
        # Evaluate the agent
        env = envs[0]  # Use only one environment for evaluation
        eval_metrics = agent.evaluate(env, num_episodes=10)
        
        print("Evaluation metrics:")
        for key, value in eval_metrics.items():
            print(f"{key}: {value}")
            
    elif args.mode == 'visualize':
        if args.checkpoint is None:
            raise ValueError("Checkpoint path must be provided for visualization")
        
        # Load checkpoint
        agent.load(args.checkpoint)
        
        # Visualize the agent
        env = envs[0]  # Use only one environment for visualization
        visualizer = Visualizer(config, env, agent)
        visualizer.visualize_episode()
    
if __name__ == '__main__':
    main()