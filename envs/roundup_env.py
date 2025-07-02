import gym
from gym import spaces
import numpy as np
import random
import logging
import os
import torch
import yaml
import ast
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from drone_navigation.algorithms.ppo_discrete_rnn.ppo_discrete_rnn import PPO_discrete_RNN
from sim.roundup_world import World
from envs.wrappers.ma_reward_wrapper import DistanceReward, AngleReward, ObstacleClearanceReward
from gymnasium.spaces import Box
from trajectory_prdiction.algorithms.transformer_predictor import TransformerPredictor


class RoundupEnv(gym.Env):
    """
    Multi-agent environment for 3 chasers and 1 runner, where chasers try to catch the runner. 
    Agents are rewarded based on their proximity to the runner or landmarks.
    """

    def __init__(self, env_params):
        """
        Initializes the environment with agent configurations, simulation, action and observation spaces, and rewards.
        """
        super().__init__()
        self.env_params = env_params
        self.num_chasers = self.env_params['drone']['num_chaser']
        self.num_runners = self.env_params['drone']['num_runner']

        self._init_simulation()        
        # Initialize simulation, action, observation spaces, and rewards
        self._init_action_spaces()
        self._init_obs_spaces()
        self._init_reward()
        # Load PPO model
        self._load_navigation_policy()
        self._load_trajectory_predictor()

        self.reset()

    def _init_simulation(self):
        """
        Initialize the simulation environment based on provided parameters.
        """
        scene_region = self.env_params['scene']['region']
        obstacle_params = self.env_params['scene']['obstacle']
        drone_params = self.env_params['drone']
        scene_type = self.env_params['scene'].get('type', 'random')
        voxel_size = self.env_params['scene'].get('voxel', {}).get('size', None)
        building_path = self.env_params.get('world', {}).get('building_path', '')

        # Create the world
        self.sim = World(
            use_gui=self.env_params['use_gui'],
            scene_type=scene_type,
            scene_region=scene_region,
            obstacle_params=obstacle_params,
            drone_params=drone_params,
            voxel_size=voxel_size,
            building_path=building_path
        )

    def _init_action_spaces(self):
        """
        Initializes action spaces for agents (either discrete or continuous).
        """
        mode = self.env_params["action"]["type"]
        if mode == 'encirclement_point':
            low = np.array([-1/8*np.pi, -1/8*np.pi], dtype=np.float32)
            high = np.array([1/8*np.pi, 1/8*np.pi], dtype=np.float32)
            self.action_spaces = {f"agent_{i}": Box(low, high, dtype=np.float32) for i in range(self.num_chasers)}

    def _init_obs_spaces(self):
        """
        Initializes observation spaces for each agent based on configuration.
        """
        features = self.env_params['observation']['features']
        num_chaser = self.env_params['drone']['num_chaser']
        obs_dim = 0

        if "target_ralative" in features:
            obs_dim += 3  # r, theta, phi

        if "runner_velocity" in features:
            obs_dim += 3  # speed, theta, phi

        if "teammate_ralative" in features:
            obs_dim += 3 * (num_chaser - 1)  # teammates (r, theta, phi)

        if "depth_image_runner" in features:
            obs_dim += self.env_params["observation"]["dim"]
 
        self.observation_spaces = {
            f"agent_{i}": spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for i in range(self.num_chasers)
        }

    def _init_reward(self):
        """
        Initializes reward components based on configuration.
        """
        self.reward_components = []
        reward_params = self.env_params["reward"]
        active = reward_params["active_components"]  # dict: {name: weight}

        if "distance_reward" in active:
            self.reward_components.append(
                DistanceReward("distance_reward", active["distance_reward"])
            )

        if "angle_reward" in active:
            self.reward_components.append(
                AngleReward("angle_reward", active["angle_reward"])
            )
        
        if "obstacle_clearance_reward" in active:
            self.reward_components.append(
                ObstacleClearanceReward("obstacle_clearance_reward", active["obstacle_clearance_reward"])
            )

    def _load_navigation_policy(self):
        """
        Loads the PPO model and its weights.
        """
        folder_path = "drone_navigation"
        env_config_path = os.path.join(folder_path, "configs", "env_config.yaml")
        ppo_config_path = os.path.join(folder_path, "configs", "ppo_config.yaml")
        model_path = os.path.join(folder_path, "models", "final_model.zip")
        with open(ppo_config_path, "r") as f:
            ppo_config = yaml.safe_load(f)
        args = argparse.Namespace(**ppo_config)
        args.state_dim = 25
        args.action_dim = 9
        args.episode_limit = 400

        self.navigation_policy = PPO_discrete_RNN(args)
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        self.navigation_policy.ac.load_state_dict(checkpoint["ac"])
        self.navigation_policy.ac.eval()
        self.navigation_policy.reset_rnn_hidden()

        with open(env_config_path, "r") as f:
            self.low_env_config = yaml.safe_load(f)

    def reset(self):
        """
        Resets the environment for a new episode and returns the initial observation.
        """
        logging.info("Resetting simulation environment ...")
        self.sim.reset()
        self.step_count = 0

        self.agents = {f"agent_{i}": self.sim.chasers[i] for i in range(self.num_chasers)}
        self.enemies = {f"enemy_{i}": self.sim.runners[i] for i in range(self.num_runners)}
        for runner in self.enemies.values():
            runner.target_position = [
                250,
                0,
                100
            ] 

        # Initialize tracking for episode rewards
        self.episode_total_reward = 0
        self.episode_component_rewards = {comp.name: 0.0 for comp in self.reward_components}

        # 初始化：在循环外执行一次
        self.runner_vels = {idx: [] for idx in self.enemies.keys()}

        obs = self.get_obs()

        return obs

    def step(self, actions):
        """
        Advances the simulation by one step based on the actions taken by the agents.
        """
        self.step_count += 1
        chaser_velocities = []
        runner_velocities = []
        # Initialize per-agent status for terminated, truncated, and collided
        agents_reached_time_limit = {idx: False for idx in self.agents.keys()}
        agents_collided = {idx: False for idx in self.agents.keys()}
        agents_captured = {idx: False for idx in self.agents.keys()}

        # obs = self.get_obs()
        for idx, chaser in self.agents.items():
            chaser.target_position = self.compute_position_from_action(idx, actions[idx])
        
        # Repeat for 4 steps 0.5s
        for _ in range(4):
            # Compute chaser velocities
            for idx, chaser in self.agents.items():
                local_obs = self.get_local_obs(chaser)
                nav_action, _ = self.navigation_policy.choose_action(local_obs, evaluate=True)
                chaser_velocities.append(self.compute_velocity_from_action(chaser, nav_action))

            # Compute runner velocities
            for idx, runner in self.enemies.items():
                local_obs = self.get_local_obs(runner)
                nav_action, _ = self.navigation_policy.choose_action(local_obs, evaluate=True)
                runner_velocities.append(self.compute_velocity_from_action(runner, nav_action))

            self.sim.step(chaser_velocities, runner_velocities, num_repeats=30)

            # 记录每个 runner 的真实速度
            for idx, runner in self.enemies.items():
                self.runner_vels[idx].append(runner.state.linear_velocity)

            # Iterate over each drone in the simulation
            for idx, drone in self.agents.items():
                # Check if the drone has captured the runner
                captured = self.check_capture(drone, self.sim.runners[0])  # Assuming runner is the first one
                
                # Check if the drone has collided (you would need a method for this)
                collided = drone.state.collided
                
                # Check if the drone has reached the time limit
                reached_time_limit = self.step_count >= self.env_params['episode']['max_episode_timesteps']
                
                # Store the results in the status dictionaries for this drone
                agents_captured[idx] = captured
                agents_collided[idx] = collided
                agents_reached_time_limit[idx] = reached_time_limit
        
        # Calculate reward
        total_reward, component_rewards = self.get_rewards()
        
        next_obs = self.get_obs()

        # Information to return
        info = {
            "total_reward": total_reward,
            "component_rewards": component_rewards,
            "captured": agents_captured,
            "reached_time_limit": agents_reached_time_limit,
            "collided": agents_collided,
        }

        return next_obs, total_reward, agents_captured, agents_reached_time_limit, info

    def get_obs(self):
        """
        Returns the observation for each agent, which includes positions, velocities, and other relevant features.
        """
        features = self.env_params['observation']['features']
        # Initialize observations as a dictionary
        obs = {}
        
        # Access runner state
        runner = self.sim.runners[0]
        runner_pos, runner_vel = runner.state.position, runner.state.linear_velocity

        for chaser_id, chaser in self.agents.items():
            sub_obs = []  # 用来存放当前 chaser 的各个子特征

            if "depth_image_runner" in features:
                # 根据 chaser 的编号,设定不同的偏移角度
                if chaser_id == "agent_0":
                    offset_angle = 0.0      # Chaser 0：原始水平速度方向
                elif chaser_id == "agent_1":
                    offset_angle = 2.0 * np.pi / 3.0    # Chaser 1：水平速度方向的左侧
                elif chaser_id == "agent_2":
                    offset_angle = -2.0 * np.pi / 3.0    # Chaser 2：水平速度方向的右侧
                else:
                    raise ValueError(f"Unknown chaser_id: {chaser_id}")

                # 调用通用函数,得到 (theta, phi)
                phi = self.get_obs_direction(runner_vel, offset_angle)

                depth_image = runner.get_depth_image_at_angle(angle_rad=phi)
                if "grid_shape" in self.env_params.get("observation", {}):
                    grid_shape = self.env_params["observation"]["grid_shape"]
                    grid_shape_tuple = ast.literal_eval(grid_shape)
                else:
                    grid_shape_tuple = (4,4)
                pooled_depth_image = self.pool_depth_image(depth_image, grid_shape_tuple)
                pooled_depth_image = pooled_depth_image.flatten()

                sub_obs.append(pooled_depth_image)

                # # 获取当前时间到秒，格式化
                # now = datetime.now()
                # timestamp = now.strftime("%Y%m%d_%H%M%S")

                # # 指定要保存的文件夹
                # save_dir = "./saved_depths"
                # # 如果该目录不存在，就先创建
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)

                # # 构造完整文件路径
                # filename = os.path.join(save_dir, f"depth_map_{timestamp}.png")

                # # 绘制并保存
                # plt.figure(figsize=(5, 5))
                # plt.imshow(depth_image, cmap='gray')
                # plt.colorbar(label='Depth (normalized)')
                # plt.title('Depth Map Visualization')
                # plt.axis('off')
                # plt.savefig(filename, bbox_inches='tight')
                # plt.close()

                # # 打印信息
                # print(f"Saved depth map at: {filename} (timestamp: {timestamp})")

            # 最终把所有 sub_obs 中的 numpy 数组合并成一个一维向量：
            if sub_obs:
                obs[f"{chaser_id}"] = np.concatenate(sub_obs)
            else:
                raise ValueError(f"[get_obs] obs_i is empty for drone {chaser_id}. Check enabled features.")

        return obs

    def get_obs_direction(self, vel_3d: np.ndarray, offset_rad: float = 0.0) -> float:
        """
        通用函数：给定一个 3D 速度向量 vel_3d(形如 [vx, vy, vz]),
        先将其投影到水平面(只保留 vx, vy),然后按 offset_rad(弧度)做旋转,
        最终返回该旋转后向量在水平面内的方位角 phi(范围 (−π, π])。

        Args:
            vel_3d (np.ndarray): 长度 3 的速度向量 [vx, vy, vz]。
            offset_rad (float): 水平平面内要对 (vx, vy) 做的旋转偏移角度(单位：弧度)。
                                正值表示向量逆时针旋转；负值表示顺时针旋转。

        Returns:
            float: phi = arctan2(rot_y, rot_x),即旋转后单位水平向量在 xy 平面内的方位角。
                取值范围在 (−π, π]。
        """
        # 1. 提取水平分量 (vx, vy)
        horiz = np.array([vel_3d[0], vel_3d[1]], dtype=np.float64)
        norm_h = np.linalg.norm(horiz)

        # 2. 水平速度过小时的兜底：默认指向 x 正方向
        if norm_h < 1e-8:
            hdir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            hdir = horiz / norm_h

        # 3. 在水平面内旋转 offset_rad(弧度)
        c = np.cos(offset_rad)
        s = np.sin(offset_rad)
        rot_x = c * hdir[0] - s * hdir[1]
        rot_y = s * hdir[0] + c * hdir[1]
        # 此时 [rot_x, rot_y] 依然是单位向量

        # 4. 计算并返回水平平面内的方位角 phi
        phi = np.arctan2(rot_y, rot_x)
        return float(phi)

    def get_rewards(self):
        """
        Computes the total reward for each drone and the component rewards.
        
        Returns:
        - total_rewards: A dictionary where the key is the drone's name, and the value is its total reward.
        - component_rewards: A dictionary where the key is the drone's name, and the value is another dictionary of component rewards for that drone.
        """
        total_rewards = {}
        component_rewards = {}
        runner = self.sim.runners[0]
        for drone_name, chaser in self.agents.items():  # Loop over each drone
            total_reward = 0
            component_rewards_for_drone = {}

            for component in self.reward_components:  # Loop over reward components
                component_reward = component.compute(chaser, runner)
                total_reward += component_reward
                component_rewards_for_drone[component.name] = component_reward

            # Store the total reward and component rewards for this drone
            total_rewards[drone_name] = total_reward
            component_rewards[drone_name] = component_rewards_for_drone

        return total_rewards, component_rewards

    def check_capture(self, drone, runner):
        """
        Checks if a specific agent (drone) has successfully captured the runner.
        
        Parameters:
        - drone: The specific drone object that we are checking for capture.
        - runner: The runner object that the drone is trying to capture.

        Returns:
        - True if the drone has successfully captured the runner, otherwise False.
        """
        # Get the capture radius from the environment parameters
        capture_radius = self.env_params['episode']['capture_radius']
        
        # Get the runner's position (assumed to be a numpy array or a tuple of coordinates)
        O = np.array(runner.state.position)  # Runner's position
        
        # Get the drone's position (also assumed to be a numpy array or a tuple of coordinates)
        D = np.array(drone.state.position)  # Drone's position

        # Calculate the distance between the drone and the runner
        distance = np.linalg.norm(D - O)

        # Check if the drone is within the capture radius of the runner
        if distance <= capture_radius:
            # If within capture radius, return True (drone has captured the runner)
            return True
        else:
            # If not within capture radius, return False (drone has not captured the runner)
            return False
  
    def get_local_obs(self, chaser):
        """
        获取当前无人机的动态观测,根据需求选择观测特征。
        
        返回：
            np.array: 拼接后的观测数据
        """

        # 获取深度图信息(前方障碍物距离)每个像素是一个浮点数,介于 [0,1] 之间
        # 靠近相机的物体 → 深度值接近0
        # 远离相机的物体 → 深度值接近1
        # 如果看向空无一物的地方,深度值趋近于 far
        # 是二维矩阵,比如 shape = (240, 320)
        depth_image = chaser.get_depth_image()
        if "grid_shape" in self.low_env_config.get("observation", {}):
            grid_shape = self.low_env_config["observation"]["grid_shape"]
            grid_shape_tuple = ast.literal_eval(grid_shape)
        else:
            grid_shape_tuple = (4,4)
        obs = self.pool_depth_image(depth_image, grid_shape_tuple)
        flatten_obs = obs.flatten()

        return flatten_obs

    def pool_depth_image(self, depth_image, grid_shape=(4, 4)):
        """
        对深度图进行最小池化,按网格划分。
        参数:
            depth_image: np.ndarray, shape=(H, W)
            grid_shape: tuple, (rows, cols)
        返回:
            pooled: np.ndarray, shape=(rows, cols)
        """
        assert isinstance(depth_image, np.ndarray), "Input must be a NumPy array"
        assert depth_image.ndim == 2, f"Expected 2D array, got {depth_image.shape}"

        H, W = depth_image.shape
        rows, cols = grid_shape
        h_step, w_step = H // rows, W // cols

        pooled = np.empty((rows, cols), dtype=depth_image.dtype)

        for i in range(rows):
            for j in range(cols):
                region = depth_image[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                pooled[i, j] = np.min(region)

        return pooled
    
    def compute_velocity_from_action(self, agent, action: np.ndarray):
        
        angle_range = 1/8
        # 定义离散动作映射表
        angle_options = [-angle_range*np.pi, 0.0, angle_range*np.pi]
        local_action_idx = [
            (dx, dy)
            for dx in angle_options
            for dy in angle_options
        ]

        # 参数
        v_horiz = 15.0  # 水平速度基准
        v_vert_max = 5.0  # 垂直速度上限
        horizontal_threshold = 2.0  # 水平距离小于该值认为已对准目标水平位置

        delta_theta, delta_phi = local_action_idx[action]
        # 获取当前和目标位置
        current_position = np.array(agent.state.position)
        target_position = np.array(agent.target_position)
        direction_vector = target_position - current_position
        norm = np.linalg.norm(direction_vector)

        # 特殊情况处理
        if norm < 1e-3:
            theta = np.pi / 2
            phi = 0.0
        else:
            theta = np.arccos(direction_vector[2] / norm)
            phi = np.arctan2(direction_vector[1], direction_vector[0])

        theta_new = np.clip(theta + delta_theta, 0, np.pi)
        phi_new = phi + delta_phi

        # 方向单位向量
        vx_unit = np.sin(theta_new) * np.cos(phi_new)
        vy_unit = np.sin(theta_new) * np.sin(phi_new)
        vz_unit = np.cos(theta_new)
        dir_unit = np.array([vx_unit, vy_unit, vz_unit], dtype=np.float32)

        # 计算水平距离
        horizontal_dist = np.linalg.norm(direction_vector[:2])

        if horizontal_dist < horizontal_threshold:
            # === 进入垂直调整阶段：以 vz 为基准 ===
            vz_target = v_vert_max * np.sign(direction_vector[2])

            # 单位方向向量
            dir_unit = np.array([vx_unit, vy_unit, vz_unit], dtype=np.float32)

            if abs(dir_unit[2]) > 1e-6:
                scale = abs(vz_target / dir_unit[2])  # 以 vz 固定为目标,缩放整向量
                v_raw = dir_unit * scale
                v_raw[2] = vz_target  # 强制精确垂直速度
            else:
                v_raw = np.array([0.0, 0.0, vz_target], dtype=np.float32)

            new_velocity = v_raw.astype(np.float32)
        else:
            # === 正常阶段：以水平速度为基准 ===
            horiz_norm = np.linalg.norm([vx_unit, vy_unit])
            if horiz_norm < 1e-6:
                vx = 0.0
                vy = 0.0
            else:
                vx = v_horiz * (vx_unit / horiz_norm)
                vy = v_horiz * (vy_unit / horiz_norm)

            vz = vz_unit * v_horiz  # 初始 vz,等比例

            # 限制 vz 不超过最大值
            if abs(vz) > v_vert_max:
                vz = v_vert_max * np.sign(vz)

            new_velocity = np.array([vx, vy, vz], dtype=np.float32)

        return new_velocity
    
    def _build_plane_basis_containing_velocity(self, v):
        """
        构造“包含速度方向 v” 的平面内的正交单位基 (e_v, e_perp, e_norm),
        其中：
          - e_v：速度方向单位向量 = v / ||v||。
          - e_perp：在“速度-竖直平面”中,与 e_v 垂直的单位向量(如果 v 与全局竖直 [0,0,1] 平行,则退回到水平 X 轴)。
          - e_norm：平面法向 = e_v × e_perp。

        参数：
        - v: 形如 [v_x, v_y, v_z] 的 3D 向量(列表或 np.array),表示目标速度向量

        返回：
        - e_v:     np.array, 3 维单位向量,方向与 v 相同
        - e_perp:  np.array, 3 维单位向量,满足 e_perp ⟂ e_v,且在“e_v-竖直”平面内
        - e_norm:  np.array, 3 维单位向量,满足 e_norm ⟂ e_v 且 ⟂ e_perp(法向量)
        """
        V = np.array(v, dtype=float)
        norm_v = np.linalg.norm(V)
        # 速度过小时的兜底，默认方向设为 [1.0, 0.0, 0.0]
        if norm_v < 1e-8:
            e_v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            e_v = V / norm_v  # 单位化

        # 2. 用全局竖直向量 [0,0,1] 在垂直于 e_v 的平面上做投影,获得 e_perp
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        # 检查 e_v 是否与竖直平行(或反平行)
        if np.allclose(np.abs(np.dot(e_v, z_axis)), 1.0, atol=1e-6):
            # 速度方向恰好竖直,则退回到用 X 轴作为第二向量
            e_perp = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            # 否则,对竖直向量做投影：e_proj = z_axis - (z_axis·e_v) * e_v
            e_proj = z_axis - np.dot(z_axis, e_v) * e_v
            e_perp = e_proj / np.linalg.norm(e_proj)

        # 3. 用 e_norm = e_v × e_perp 得到平面法向
        e_norm = np.cross(e_v, e_perp)
        e_norm /= np.linalg.norm(e_norm)

        return e_v, e_perp, e_norm

    def _compute_position_with_adjustments(self,
                                           target_position,
                                           velocity,
                                           radius,
                                           base_azimuth_rad,
                                           delta_azimuth_rad,
                                           delta_elevation_rad):
        """
        在“包含速度方向的平面”上,以 target_position 为圆心、radius 为半径,
        计算圆周上相对于“速度方向前方 + base_azimuth_rad + delta_azimuth_rad” 的位置,
        并再加上一个沿该平面法向的“俯仰偏移” delta_elevation_rad。

        具体步骤：
        1. 构造正交单位基 (e_v, e_perp, e_norm):
             - e_v = velocity / ||velocity||；
             - e_perp 在“速度-竖直”平面内,与 e_v 垂直；
             - e_norm = e_v × e_perp 为平面法向。
        2. 设 θ_total = base_azimuth_rad + delta_azimuth_rad；
           设 ϕ = delta_elevation_rad。
        3. 平面内“水平分量”:
             horizontal_dir = cos(ϕ) * [cos(θ_total)*e_v + sin(θ_total)*e_perp]
        4. 沿法向“垂直分量”:
             vertical_dir = sin(ϕ) * e_norm
        5. 合并后得到单位方向向量:
             dir_vec = horizontal_dir + vertical_dir
             dir_vec /= ||dir_vec||
        6. 最终位置 P = target_position + radius * dir_vec

        参数：
        - target_position:      np.array 或 list,形如 [T_x, T_y, T_z]
        - velocity:             np.array 或 list,形如 [v_x, v_y, v_z]
        - radius:               float,与目标保持的距离
        - base_azimuth_rad:     float,名义方位角(0、+2π/3、-2π/3)
        - delta_azimuth_rad:    float,方位微调(相对于 名义方位),弧度制
        - delta_elevation_rad:  float,俯仰微调(相对于平面),弧度制

        返回：
        - np.array, 长度为 3,UAV 的最终三维坐标 [x, y, z]
        """
        T = np.array(target_position, dtype=float)
        V = np.array(velocity, dtype=float)

        # 1. 构造三个正交单位向量：e_v, e_perp, e_norm
        e_v, e_perp, e_norm = self._build_plane_basis_containing_velocity(V)

        # 2. 计算“总方位角” θ = base_azimuth_rad + delta_azimuth_rad
        θ_total = base_azimuth_rad + delta_azimuth_rad
        # 3. 计算“俯仰角” ϕ = delta_elevation_rad
        ϕ = delta_elevation_rad

        # 4. 计算平面内“水平分量”
        #    先算出平面内纯水平方向：dir_plane = cos(θ_total)*e_v + sin(θ_total)*e_perp
        dir_plane = np.cos(θ_total) * e_v + np.sin(θ_total) * e_perp
        #    再乘以 cos(ϕ) 得到投影到该平面里的分量
        horizontal_dir = np.cos(ϕ) * dir_plane

        # 5. 计算沿法向的分量：vertical_dir
        vertical_dir = np.sin(ϕ) * e_norm

        # 6. 合并并归一化：
        dir_vec = horizontal_dir + vertical_dir
        dir_vec /= np.linalg.norm(dir_vec)

        # 7. 最终位置
        P = T + radius * dir_vec
        return P
    
    def compute_intercept_point(self, escapee_pos, escapee_vel, pursuer_pos, max_speed=5.0):
        """
        最快拦截（同速）计算，基于逃跑者路径与追击者之间的垂直平分线交点。
        ----------
        escapee_pos : ndarray (2 or 3) 逃跑者当前坐标
        escapee_vel : ndarray (2 or 3) 逃跑者的速度方向
        pursuer_pos : ndarray (2 or 3) 追击者当前坐标
        max_speed    : float            双方最大速度模长(默认 15)
        ----------
        返回 intercept_point；若无可行拦截则返回 None
        """
        
        # 单位化逃跑者速度向量
        escapee_vel = escapee_vel / np.linalg.norm(escapee_vel)
        
        # 计算追击者与逃跑者之间的初始距离
        d = escapee_pos - pursuer_pos
        
        # 计算追击者与逃跑者之间的中点
        midpoint = (escapee_pos + pursuer_pos) / 2.0  
        
        # 计算追击者与逃跑者之间的连线方向，获取垂直方向（即正交向量）
        # 对于二维，旋转90度，得到垂直方向
        normal_vector = np.array([-d[1], d[0]])  # 2D case (rotating by 90 degrees)
        
        # 计算逃跑者路径的参数化表示：逃跑者位置 + t * 逃跑者速度
        # 逃跑者路径公式：p(t) = escapee_pos + t * escapee_vel
        
        # 设垂直平分线为：l(s) = midpoint + s * normal_vector
        # 我们需要求解 s 和 t 使得这两条线相交
        
        # 使用线性方程组来计算 t 和 s
        A = np.vstack([escapee_vel, normal_vector]).T  # 系数矩阵
        b = midpoint - escapee_pos  # 方程的右侧
        
        try:
            # 通过求解线性方程组，得到 t 和 s
            t_s = np.linalg.solve(A, b)
            t = t_s[0]  # 逃跑者的参数 t
            
            # 计算拦截点
            intercept_point = escapee_pos + t * escapee_vel
            
            return intercept_point
        except np.linalg.LinAlgError:
            return None  # 如果无法求解，返回 None
    
    def compute_position_from_action(self, 
                                     chaser_id, 
                                     action):
        """
        计算指定 UAV(chaser_0, chaser_1, chaser_2)的三维位置,
        使其均匀分布在以目标为圆心、包含速度方向的圆周上,并支持在圆周位置上微调：
          - cloud_base_azimuth = { chaser_0: 0, chaser_1: +2π/3, chaser_2: -2π/3 }
          - action = [delta_azimuth_rad, delta_elevation_rad],可选；如果传 None,则取为 [0, 0]。

        参数：
        - chaser_id:           str,必须是 'chaser_0'、'chaser_1' 或 'chaser_2'
        - action:              list 或 tuple,形如 [delta_azimuth_rad, delta_elevation_rad]
                               如果传 None 或长度不对,就当作 [0,0] 处理。

        返回：
        - np.array, 长度为 3,该 UAV 的最终三维坐标
        """
        # 1. 获取目标当前位置与速度向量
        runner = self.enemies["enemy_0"]
        chaser = self.agents[chaser_id]
        L_in = 8
        num_iters = 1
        for idx, runner in self.enemies.items():
            runner_pos = runner.state.position
            vel_hist = self.runner_vels[idx]
            if len(vel_hist) < L_in:
                target_velocity = runner.state.linear_velocity
                target_position = runner_pos
            else:
                buf = vel_hist[-L_in:].copy()
                for _ in range(num_iters):
                    v_pred = self.trajectory_predictor.predict(torch.from_numpy(np.array(buf[-L_in:])).float().unsqueeze(0)).squeeze(0).cpu().numpy()
                    buf.extend(v_pred)
                # v_pred = np.array([2, 1, 0])
                # v_pred =  [15 * v_pred / np.linalg.norm(v_pred)]
                runner_pos = runner.state.position
                chaser_pos = chaser.state.position
                v = v_pred[-1][:2]
                intercept_point = runner_pos[:2] + v * 10
                if np.linalg.norm(runner_pos-chaser_pos) < 150:
                    intercept_point = runner_pos[:2] + v * 7
                if np.linalg.norm(runner_pos-chaser_pos) < 100:
                    intercept_point = runner_pos[:2] + v * 5
                if np.linalg.norm(runner_pos-chaser_pos) < 75:
                    intercept_point = runner_pos[:2] + v * 3
                if np.linalg.norm(runner_pos-chaser_pos) < 45:
                    intercept_point = runner_pos[:2] + v * 1
                if np.linalg.norm(runner_pos-chaser_pos) < 15:
                    intercept_point = runner_pos[:2] + v * 0.5
                # 将垂直方向的 Z 值保持不变，构造新的位置
                target_position = np.append(intercept_point, runner_pos[2])  # 保持 Z 不变
                target_velocity = v_pred[-1]

        # 2. UAV 与目标保持的距离 radius
        radius = 20.0  # 这里可以根据实际需求修改

        # 3. 根据 chaser_id 确定名义方位 base_azimuth_rad
        if chaser_id == "agent_0":
            radius = 1
            base_azimuth_rad = 0.0
        elif chaser_id == "agent_1":
            base_azimuth_rad = np.pi / 3
        elif chaser_id == "agent_2":
            base_azimuth_rad = -np.pi / 3
        else:
            raise ValueError(f"Unknown chaser_id: {chaser_id}")

        delta_azimuth_rad,delta_elevation_rad = action

        # 4. 调用 _compute_position_with_adjustments 计算最终位置
        position = self._compute_position_with_adjustments(
            target_position=target_position,
            velocity=target_velocity,
            radius=radius,
            base_azimuth_rad=base_azimuth_rad,
            delta_azimuth_rad=delta_azimuth_rad,
            delta_elevation_rad=delta_elevation_rad
        )
        return position
    
    def _load_trajectory_predictor(self):
        # ------------------ 1. 配置参数 ------------------
        model_path = "trajectory_prdiction/models/model.pth"
        config_path = "trajectory_prdiction/configs/config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        input_dim = config.get("input_dim", 3)
        embed_dim = config["embed_dim"]
        n_heads   = config["n_heads"]
        num_layers= config["num_layers"]
        dropout   = config["dropout"]
        L_in      = config["L_in"]
        L_out     = config["L_out"]

        self.trajectory_predictor = TransformerPredictor(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_heads=n_heads,
            num_layers=num_layers,
            dropout=dropout,
            L_in=L_in,
            L_out=L_out,
            device="cuda:0",  # 传入 "cuda" 或 "cpu"
        )

        self.trajectory_predictor.load_model(model_path)
        print(f"Model weights loaded from → {model_path}")