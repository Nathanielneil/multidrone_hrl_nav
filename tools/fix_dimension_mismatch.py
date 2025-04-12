import os
import re

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, '..')

# 1. 修复 config.py 中的观察空间维度
def fix_config():
    config_path = os.path.join(base_dir, 'config', 'config.py')
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 将观察空间维度从 12 更改为 13
    content = re.sub(r'self\.observation_space_dim\s*=\s*12', 'self.observation_space_dim = 13', content)
    
    with open(config_path, 'w') as f:
        f.write(content)
    
    print("已修复 config.py 中的观察空间维度")

# 2. 在 hierarchical_agent.py 中添加额外的错误检查
def fix_hierarchical_agent():
    agent_path = os.path.join(base_dir, 'models', 'hierarchical_agent.py')
    with open(agent_path, 'r') as f:
        content = f.read()
    
    # 在 act 方法中添加维度检查
    pattern = r'def act\(self, observation, deterministic=False\):\s*"""'
    replacement = r'def act(self, observation, deterministic=False):\n        """\n        # 调试信息 - 检查维度\n        print(f"State shape: {observation[\'states\'].shape}")\n        print(f"Expected state dim: {self.config.observation_space_dim * self.num_drones}")\n        '
    
    content = re.sub(pattern, replacement, content)
    
    # 修复状态分割逻辑
    pattern = r'individual_states = \[\]\s+for i in range\(self\.num_drones\):'
    replacement = r'individual_states = []\n        single_obs_dim = self.config.observation_space_dim\n        \n        # 检查维度是否正确\n        if len(state_obs) != self.num_drones * single_obs_dim:\n            print(f"警告: 状态维度不匹配! 期望 {self.num_drones * single_obs_dim}，实际 {len(state_obs)}")\n        \n        for i in range(self.num_drones):'
    
    content = re.sub(pattern, replacement, content)
    
    # 修复索引计算
    pattern = r'start_idx = i \* single_obs_dim\s+end_idx = start_idx \+ single_obs_dim\s+individual_states\.append\(state_obs\[start_idx:end_idx\]\)'
    replacement = r'start_idx = i * single_obs_dim\n            end_idx = start_idx + single_obs_dim\n            # 添加范围检查\n            if end_idx <= len(state_obs):\n                individual_states.append(state_obs[start_idx:end_idx])\n            else:\n                # 如果索引超出范围，使用零向量\n                individual_states.append(torch.zeros(single_obs_dim, device=self.device))'
    
    content = re.sub(pattern, replacement, content)
    
    with open(agent_path, 'w') as f:
        f.write(content)
    
    print("已修复 hierarchical_agent.py 中的维度检查和状态分割逻辑")

# 3. 修复 high_level_policy.py 中的图像处理
def fix_high_level_policy():
    policy_path = os.path.join(base_dir, 'models', 'high_level_policy.py')
    with open(policy_path, 'r') as f:
        content = f.read()
    
    # 修复 forward 方法中的图像处理逻辑
    pattern = r'if image\.dim\(\) == 4:'
    replacement = r'if image.dim() == 4:  # 形状为 [num_drones, channels, height, width]'
    
    content = re.sub(pattern, replacement, content)
    
    with open(policy_path, 'w') as f:
        f.write(content)
    
    print("已修复 high_level_policy.py 中的图像处理逻辑")

# 执行所有修复
def apply_all_fixes():
    fix_config()
    fix_hierarchical_agent()
    fix_high_level_policy()
    print("所有修复已应用，请尝试使用较小的目标位置运行程序：")
    print("python main.py --mode train --target_pos 50.0 50.0 -10.0 --num_envs 4")

if __name__ == "__main__":
    apply_all_fixes()