import os
import yaml
import time
import torch
from datetime import timedelta
# 1. 导入 TensorBoard 的 SummaryWriter
from torch.utils.tensorboard import SummaryWriter


from envs.roundup_env import RoundupEnv
from algorithms.maddpg_continous.main_parameters import main_parameters
from algorithms.maddpg_continous.utils.runner import RUNNER
from algorithms.maddpg_continous.agents.maddpg.MADDPG_agent import MADDPG


def get_env(env_params):
    """Create environment and get observation and action dimension of each agent in this environment."""
    env = RoundupEnv(env_params)
    env.reset()

    _dim_info = {}
    action_bound = {}

    for agent_id in env.agents.keys():
        _dim_info[agent_id] = []      # [obs_dim, act_dim]
        action_bound[agent_id] = []   # [low, high]

        _dim_info[agent_id].append(env.observation_spaces[agent_id].shape[0])
        _dim_info[agent_id].append(env.action_spaces[agent_id].shape[0])

        action_bound[agent_id].append(env.action_spaces[agent_id].low)
        action_bound[agent_id].append(env.action_spaces[agent_id].high)

    return env, _dim_info, action_bound


if __name__ == '__main__':
    # ——————————————————————————————————————————
    # Step 1: 选择设备（GPU/CPU）并打印
    # ——————————————————————————————————————————
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ——————————————————————————————————————————
    # Step 2: 生成一个 run_name，用于组织日志文件夹
    # ——————————————————————————————————————————
    start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_name = f"maddpg_{start_time_str}"

    # ——————————————————————————————————————————
    # Step 3: 初始化 TensorBoard 的 SummaryWriter
    #    我们把日志统一写到：logs/tensorboard/<run_name> 下
    # ——————————————————————————————————————————
    tb_log_dir = os.path.join("logs", "tensorboard", run_name)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    # ——————————————————————————————————————————
    # Step 4: 加载或生成超参数（main_parameters）
    # ——————————————————————————————————————————
    args = main_parameters()

    # ——————————————————————————————————————————
    # Step 5: 定义模型 checkpoint 目录 & config 目录
    # ——————————————————————————————————————————
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, f'logs/maddpg/{run_name}')
    chkpt_dir =  os.path.join(log_dir, 'model')
    config_dir = os.path.join(log_dir, 'config')

    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(chkpt_dir, exist_ok=True)

    args.chkpt_dir = chkpt_dir

    # ——————————————————————————————————————————
    # Step 6: 将超参数写到 YAML 文件里，以保证可复现性
    # ——————————————————————————————————————————
    args_save_path = os.path.join(config_dir, 'maddpg-config.yaml')
    with open(args_save_path, 'w') as f:
        yaml.dump(vars(args), f)

    # ——————————————————————————————————————————
    # Step 7: 把超参数记录到 TensorBoard——使用 add_hparams
    #    TensorBoard 会在 “HParams (Hyperparameters)” 标签页里展示
    #    注意：add_hparams 的格式是：{k: v} 键值对，以及一个 metrics 字典
    #    metrics dict 可以暂时写成空 {}，看后续训练时再 add_scalar 进行记录
    # ——————————————————————————————————————————
    hparam_dict = vars(args)
    # 由于 add_hparams 要求 metrics 里至少要给一个数字，这里先给个 0，
    # 后面真正训练时每个指标会独立记录。
    dummy_metric = {"dummy_metric": 0}
    writer.add_hparams(hparam_dict, dummy_metric)

    # ——————————————————————————————————————————
    # Step 8: 读取环境配置，并存到 config 目录
    # ——————————————————————————————————————————
    env_config_path = "configs/env_config.yaml"
    with open(env_config_path, "r") as f:
        env_config = yaml.safe_load(f)

    env_config_save_path = os.path.join(config_dir, 'env_config.yaml')
    with open(env_config_save_path, 'w') as f:
        yaml.dump(env_config, f)

    # ——————————————————————————————————————————
    # Step 9: 初始化环境，获取维度信息
    # ——————————————————————————————————————————
    env, dim_info, action_bound = get_env(env_config)

    # ——————————————————————————————————————————
    # Step 10: 创建 MADDPG Agent
    # ——————————————————————————————————————————
    agent = MADDPG(
        dim_info,
        args.buffer_capacity,
        args.batch_size,
        args.actor_lr,
        args.critic_lr,
        action_bound,
        _chkpt_dir=chkpt_dir,
        _device=device
    )

    # ——————————————————————————————————————————
    # Step 11: 创建 Runner 并传入 writer，让它负责记录训练过程
    #    注意：我们假设在 RUNNER 定义中，支持接收一个 writer 参数
    # ——————————————————————————————————————————
    runner = RUNNER(agent, env, args, device, mode='train', tb_writer=writer)

    # ——————————————————————————————————————————
    # Step 12: 开始训练
    #    Runner 内部会调用 writer.add_scalar(...) 记录各指标
    # ——————————————————————————————————————————
    start_wall_time = time.time()
    runner.train()
    end_wall_time = time.time()

    # ——————————————————————————————————————————
    # Step 13: 打印训练耗时
    # ——————————————————————————————————————————
    elapsed_seconds = int(end_wall_time - start_wall_time)
    training_duration = str(timedelta(seconds=elapsed_seconds))
    print("\n=========== 训练完成! ===========")
    print(f"训练设备: {device}")
    print(f"训练用时: {training_duration}")

    # ——————————————————————————————————————————
    # Step 14: 保存最终模型
    # ——————————————————————————————————————————
    print("--- saving trained models ---")
    agent.save_model()
    print("--- trained models saved ---")

    # ——————————————————————————————————————————
    # Step 15: 关闭 SummaryWriter（非常重要！）
    # ——————————————————————————————————————————
    writer.close()

    print(f"你可以运行以下命令，用 TensorBoard 可视化日志：")
    print(f"tensorboard --logdir=logs/tensorboard/{run_name}")
