import torch
import numpy as np
import gym
import argparse
from drone_navigation.algorithms.ppo_discrete_rnn.normalization import Normalization, RewardScaling
from drone_navigation.algorithms.ppo_discrete_rnn.replaybuffer import ReplayBuffer
from drone_navigation.algorithms.ppo_discrete_rnn.ppo_discrete_rnn import PPO_discrete_RNN
from envs.navigation_env import NavigationEnv
import wandb
import yaml
from datetime import datetime
import os
import logging


def test(folder_path):
    # 自动识别文件路径
    env_config_path = os.path.join(folder_path, "configs", "env_config.yaml")
    ppo_config_path = os.path.join(folder_path, "configs", "ppo_config.yaml")
    model_path =  os.path.join(folder_path, "models", "final_model.zip")
    
    # 读取 YAML 配置
    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)
    with open(ppo_config_path, "r") as f:
        ppo_config = yaml.safe_load(f)

    # 转为 Namespace 格式
    args = argparse.Namespace(**ppo_config)
    # 初始化环境
    env = NavigationEnv(env_config)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # env.seed(args.seed)
    env.action_space.seed(args.seed)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.episode_limit = env._max_episode_steps


    # 状态归一化
    if args.use_state_norm:
        state_norm = Normalization(shape=args.state_dim)

    # 初始化 agent 并加载模型
    agent = PPO_discrete_RNN(args)
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    agent.ac.load_state_dict(checkpoint["ac"])
    agent.ac.eval()

    success_count = 0
    collision_count = 0
    total_reward = 0

    for i in range(100):
        s = env.reset()
        done = False
        agent.reset_rnn_hidden()
        episode_reward = 0
        while not done:
            if args.use_state_norm:
                s = state_norm(s, update=False)
            a, _ = agent.choose_action(s, evaluate=True)
            s, r, done, infos = env.step(a)
            episode_reward += r
        total_reward += episode_reward
        if infos.get("arrival", False):
            success_count += 1
        if infos.get("collision", False):
            collision_count += 1
        print(f"[Episode {i+1}] Reward: {episode_reward:.2f}, Info: {infos}")

    print("\n=== Test Summary ===")
    print(f"Model Path: {model_path}")
    print(f"Average Reward: {total_reward / 100:.2f}")
    print(f"Success Rate: {success_count / 100:.2%}")
    print(f"Collision Rate: {collision_count / 100:.2%}")
    env.close()

if __name__ == "__main__":
    test(
        folder_path="drone_navigation",
    )

