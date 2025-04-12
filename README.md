# AirSim多无人机分层强化学习自主探索系统

本项目基于AirSim-UE(虚幻引擎4.27)仿真平台，实现了3个四旋翼无人机基于分层强化学习的自主探索功能。系统支持集群朝指定目标点探索、无人机集群自主协作、GPU加速训练、多环境并行训练以及模型评估与可视化。

## 1. 项目特点

- **分层强化学习**: 采用高层策略、中层策略和低层策略三层架构
- **多无人机协作**: 支持多无人机集群协同探索
- **自主避障**: 利用相机和IMU传感器实现自主避障
- **目标导向**: 无人机集群能朝着指定目标点高效探索
- **高效训练**: 支持GPU加速和多环境并行训练
- **可视化分析**: 丰富的评估和可视化工具

## 2. 系统架构

系统采用分层强化学习架构:

1. **高层策略**: 负责集群整体策略和目标分配，决定各无人机的任务优先级和分工
2. **中层策略**: 处理单个无人机的路径规划，生成到达高层目标的具体路径点
3. **低层策略**: 控制无人机的稳定和移动，实现精确的姿态和速度控制

## 3. 环境配置

### 3.1 基础环境要求

- Ubuntu 18.04/20.04 或 Windows 10
- Python 3.8+
- CUDA 11.0+ (用于GPU加速)
- 虚幻引擎 4.27
- AirSim

### 3.2 安装步骤

#### 安装虚幻引擎4.27
从Epic Games Launcher下载并安装UE 4.27。

#### 安装AirSim
```bash
git clone https://github.com/microsoft/AirSim.git
cd AirSim
./setup.sh
./build.sh
```

#### 创建Python环境
```bash
conda create -n airsim_rl python=3.8
conda activate airsim_rl

# 安装必要的Python库
pip install torch torchvision torchaudio cudatoolkit=11.3
pip install gym stable-baselines3 msgpack-rpc-python airsim
pip install matplotlib numpy pandas tqdm opencv-python
```

### 3.3 配置AirSim

创建`settings.json`文件在`~/Documents/AirSim/`目录下：

```json
{
  "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "ViewMode": "SpringArmChase",
  "Vehicles": {
    "Drone1": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "EnableCollisionPassthrough": false,
      "EnableCollisions": true,
      "AllowAPIAlways": true,
      "RC": {
        "RemoteControlID": 0
      },
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.25, "Y": 0.0, "Z": 0.1,
          "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        }
      },
      "Sensors": {
        "Imu": {
          "SensorType": 2,
          "Enabled": true
        }
      },
      "X": 0, "Y": 0, "Z": 0,
      "Pitch": 0, "Roll": 0, "Yaw": 0
    },
    "Drone2": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "EnableCollisionPassthrough": false,
      "EnableCollisions": true,
      "AllowAPIAlways": true,
      "RC": {
        "RemoteControlID": 1
      },
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.25, "Y": 0.0, "Z": 0.1,
          "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        }
      },
      "Sensors": {
        "Imu": {
          "SensorType": 2,
          "Enabled": true
        }
      },
      "X": 5, "Y": 0, "Z": 0,
      "Pitch": 0, "Roll": 0, "Yaw": 0
    },
    "Drone3": {
      "VehicleType": "SimpleFlight",
      "DefaultVehicleState": "Armed",
      "EnableCollisionPassthrough": false,
      "EnableCollisions": true,
      "AllowAPIAlways": true,
      "RC": {
        "RemoteControlID": 2
      },
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 640,
              "Height": 480,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.25, "Y": 0.0, "Z": 0.1,
          "Pitch": 0.0, "Roll": 0.0, "Yaw": 0.0
        }
      },
      "Sensors": {
        "Imu": {
          "SensorType": 2,
          "Enabled": true
        }
      },
      "X": 0, "Y": 5, "Z": 0,
      "Pitch": 0, "Roll": 0, "Yaw": 0
    }
  }
}
```

### 3.4 安装本项目

```bash
git clone <repository-url>
cd multidrone_hrl

# 创建必要的目录
mkdir -p checkpoints logs config
```

## 4. 使用指南

### 4.1 训练无人机

```bash
# 使用默认配置进行训练
python main.py --mode train

# 指定目标位置和环境数量
python main.py --mode train --target_pos 50.0 50.0 -10.0 --num_envs 4

# 使用自定义配置文件
python main.py --mode train --config custom
```

### 4.2 评估无人机

```bash
# 使用指定检查点进行评估
python main.py --mode eval --checkpoint checkpoints/checkpoint_1000000.pt

# 指定目标位置进行评估
python main.py --mode eval --checkpoint checkpoints/checkpoint_final.pt --target_pos 40.0 40.0 -15.0
```

### 4.3 可视化无人机轨迹

```bash
# 可视化无人机轨迹
python main.py --mode visualize --checkpoint checkpoints/checkpoint_final.pt
```

## 5. 系统参数配置

系统的主要参数可以在`config/config.py`中配置，主要参数包括：

- **num_drones**: 无人机数量
- **target_pos**: 目标位置
- **target_radius**: 目标成功半径
- **max_steps_per_episode**: 每个回合的最大步数
- **learning_rate**: 学习率
- **gamma**: 折扣因子
- **batch_size**: 批量大小
- **num_envs**: 并行环境数量

## 6. 项目结构

```
multidrone_hrl/
├── config/
│   └── config.py               # 配置文件
├── environment/
│   ├── __init__.py
│   ├── airsim_env.py           # AirSim单无人机环境
│   └── multi_drone_env.py      # 多无人机环境
├── models/
│   ├── __init__.py
│   ├── hierarchical_agent.py   # 分层代理主类
│   ├── high_level_policy.py    # 高层策略
│   ├── mid_level_policy.py     # 中层策略
│   └── low_level_policy.py     # 低层策略
├── training/
│   ├── __init__.py
│   ├── distributed_trainer.py  # 分布式训练器
│   └── ppo_trainer.py          # PPO训练算法
├── utils/
│   ├── __init__.py
│   ├── data_collector.py       # 数据收集工具
│   ├── visualization.py        # 可视化工具
│   └── reward_functions.py     # 奖励函数定义
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # 评估指标
│   └── visualizer.py           # 评估可视化
└── main.py                     # 主程序入口
```

## 7. 算法详解

### 7.1 分层强化学习框架

分层强化学习框架将决策过程分为三个层次，每个层次负责不同抽象级别的任务：

1. **高层策略**:
   - 输入：全局状态、所有无人机的相机图像
   - 输出：每个无人机的全局目标
   - 作用：协调无人机群体，分配探索区域

2. **中层策略**:
   - 输入：单个无人机状态、高层目标
   - 输出：详细的路径点
   - 作用：规划路径，避开障碍物

3. **低层策略**:
   - 输入：单个无人机状态、中层路径点
   - 输出：控制命令（速度、偏航角速度）
   - 作用：精确控制无人机运动

### 7.2 多无人机协作

多无人机协作通过以下机制实现：

- **信息共享**：无人机之间共享状态信息
- **相对位置维持**：保持适当的队形距离
- **任务分配**：高层策略为每个无人机分配不同任务
- **冲突避免**：考虑无人机之间的位置关系，避免碰撞

### 7.3 奖励函数设计

系统的奖励函数包含多个组成部分：

- **目标奖励**：向目标靠近获得正奖励
- **队形奖励**：保持适当队形获得奖励
- **碰撞惩罚**：发生碰撞获得负奖励
- **效率惩罚**：每步获得小惩罚，鼓励快速完成任务
- **成功奖励**：到达目标获得大量奖励

## 8. 实验结果

在标准测试环境中，系统表现出以下性能：

- **成功率**：85%以上的无人机集群能够成功到达目标
- **平均时间**：训练后的无人机平均需要200步左右到达目标
- **避障能力**：能够自主避开95%以上的障碍物
- **队形保持**：保持良好的三角形队形结构

## 9. 故障排除

- **AirSim连接问题**：确保UE4.27环境已启动并运行AirSim插件
- **GPU内存不足**：尝试减小batch_size或image_height/width
- **训练不稳定**：调整learning_rate和gamma参数
- **无人机碰撞**：增加reward_collision_penalty参数值
- **探索效率低**：调整reward_target_factor和reward_step_penalty

## 10. 未来工作

- **支持更多种类的传感器**：如激光雷达、深度摄像头
- **动态目标追踪**：支持目标的动态变化
- **更复杂环境探索**：支持在复杂室内环境中的探索
- **自适应策略调整**：根据环境动态调整策略参数
- **多模式任务切换**：实现探索、追踪、侦察等多种任务模式
"# multidrone_hrl_nav" 
