from env.roundup_env import ThreeChaserOneRunnerEnv

from model.maddpg_continous.main_parameters import main_parameters
from model.maddpg_continous.utils.runner import RUNNER
from model.maddpg_continous.agents.maddpg.MADDPG_agent import MADDPG
import torch
import os
import yaml

import time
from datetime import timedelta

def get_env(env_params):
    """Create environment and get observation and action dimension of each agent in this environment."""
    # Initialize the environment
    env = ThreeChaserOneRunnerEnv(env_params)
    env.reset()

    # Initialize dictionaries to store dimensions and action bounds
    _dim_info = {}
    action_bound = {}

    # Iterate over all agents to gather information about their observation and action spaces
    for agent_id in env.agents.keys():
        _dim_info[agent_id] = []  # [obs_dim, act_dim]
        action_bound[agent_id] = []  # [low action, high action]

        # Get the observation and action dimensions
        _dim_info[agent_id].append(env.observation_spaces[agent_id].shape[0])
        _dim_info[agent_id].append(env.action_spaces[agent_id].shape[0])

        # Get the action bounds (low and high)
        action_bound[agent_id].append(env.action_spaces[agent_id].low)
        action_bound[agent_id].append(env.action_spaces[agent_id].high)

    # Return the environment and the gathered information
    return env, _dim_info, action_bound


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() 
    #                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    device = "cuda:1"
    print("Using device:",device)
    start_time = time.time() # 记录开始时间
    
    # 模型保存路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    chkpt_dir = os.path.join(current_dir, 'models/maddpg_models/')
    # 定义参数
    args = main_parameters()

    env_config_path = "config/env_config.yaml"
    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)
    env, dim_info, action_bound = get_env(env_config)
    # print(env, dim_info, action_bound)
    # 创建MA-DDPG智能体 dim_info: 字典，键为智能体名字 内容为二维数组 分别表示观测维度和动作维度 是观测不是状态 需要注意。
    agent = MADDPG(dim_info, args.buffer_capacity, args.batch_size, args.actor_lr, args.critic_lr, action_bound, _chkpt_dir = chkpt_dir, _device = device)
    # 创建运行对象
    runner = RUNNER(agent, env, args, device, mode = 'train')
    # 开始训练
    runner.train()
    print("agent",agent)

    # 计算训练时间
    end_time = time.time()
    training_time = end_time - start_time
    # 转换为时分秒格式
    training_duration = str(timedelta(seconds=int(training_time)))
    print(f"\n===========训练完成!===========")
    print(f"训练设备: {device}")
    print(f"训练用时: {training_duration}")

    print("--- saving trained models ---")
    agent.save_model()
    print("--- trained models saved ---")
    


