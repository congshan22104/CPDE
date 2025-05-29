"""
env/low_level_env.py
"""

import gym
from gym import spaces
import numpy as np
import random
import logging
import yaml
from sim.world import World
from env.wrappers.reward_wrapper import (
    TargetProgressReward,
    NonlinearCollisionPenalty,
    DirectionReward,
    VelocityReward,
    LinearCollisionPenalty,
    TerminalReward,
    SphericalDirectionReward,
    DistanceToObstacleReward,
    NonlinearSphericalDirectionReward,
    ImageNonlinearCollisionPenalty,
    ImageNonlinearCollisionPenalty2,
    ImageLinearCollisionPenalty,
    CosineSphericalDirectionReward,
    CosineSphericalDirectionReward2,
    InterpolationSphericalDirectionReward,
    TanhSphericalDirectionReward
)

import pybullet as p
import wandb
import ast


class NavigationEnv(gym.Env):
    def __init__(self, env_params):
        super().__init__()
        self.env_params = env_params
        self._init_simulation()
        self._init_obs_space()
        self._init_action_space()
        self._init_reward()
        self._max_episode_steps = self.env_params['episode']['max_episode_timesteps']
        self._action_repeat = self.env_params['episode']['action_repeat']

        # 初始化每个组件的回合累计奖励
        self.episode_total_reward = 0
        self.episode_component_rewards = {comp.name: 0.0 for comp in self.reward_components}
        
    def _init_simulation(self):
        scene_region = self.env_params['scene']['region']
        obstacle_params = self.env_params['scene']['obstacle']
        drone_params = self.env_params['drone']
        scene_type = self.env_params['scene'].get('type', 'random')
        voxel_size = self.env_params['scene'].get('voxel', {}).get('size', None)
        building_path = self.env_params.get('world', {}).get('building_path', '')

        self.sim = World(
            use_gui=self.env_params['use_gui'],
            scene_type=scene_type,
            scene_region=scene_region,
            obstacle_params=obstacle_params,
            drone_params=drone_params,
            voxel_size=voxel_size,
            building_path=building_path
        )

    def _init_action_space(self):
        """
        根据动作控制模式初始化动作空间。

        参数:
            mode (str): 'cartesian', 'spherical', 或 'adjust'
        """
        mode = self.env_params["action"]["type"]

        if mode == "cartesian":
            self.action_space = spaces.Box(
                low=np.array([-15.0, -15.0, -15.0], dtype=np.float32),
                high=np.array([15.0, 15.0, 15.0], dtype=np.float32),
                dtype=np.float32
            )

        elif mode == "spherical":
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -np.pi], dtype=np.float32),       # v ∈ [0, 15], θ ∈ [0, π], φ ∈ [-π, π]
                high=np.array([15.0, np.pi, np.pi], dtype=np.float32),
                dtype=np.float32
            )

        elif mode == "adjust":
            self.angle_range = eval(self.env_params["action"]["range"])
            self.action_space = spaces.Box(
                low=np.array([-self.angle_range*np.pi, -self.angle_range*np.pi], dtype=np.float32),  # v_abs, Δθ, Δφ
                high=np.array([self.angle_range*np.pi, self.angle_range*np.pi], dtype=np.float32),
                dtype=np.float32
            )
        elif mode in ["discrete_adjust", "discrete_adjust_2"]:
            self.action_space = spaces.Discrete(9)
            self.angle_range = eval(self.env_params["action"]["range"])
            # 定义离散动作映射表
            angle_options = [-self.angle_range*np.pi, 0.0, self.angle_range*np.pi]
            self.action_idx_to_delta = [
                (dx, dy)
                for dx in angle_options
                for dy in angle_options
            ]
        elif mode in ["horizon_discrete_adjust_2"]:
            self.angle_range = eval(self.env_params["action"]["range"])
            
            # # 俯仰角 delta_theta（极角），3个选项
            # theta_options = [
            #     -self.angle_range * (1/8) * np.pi,
            #     0.0,
            #     self.angle_range * (1/8) * np.pi
            # ]
            theta_options = [0.0]

            # 偏航角 delta_phi（方位角），8个选项，均匀分布在 [-1/2π, 1/2π]
            phi_options = [
                self.angle_range * (i / 7.0 - 0.5) * np.pi
                for i in range(8)
            ]

            # 构建动作映射表（27 个动作：3 × 9）
            self.action_idx_to_delta = [
                (d_theta, d_phi)
                for d_theta in theta_options
                for d_phi in phi_options
            ]

            self.action_space = spaces.Discrete(len(self.action_idx_to_delta))

        
        elif mode == "horizon_discrete_adjust_3":
            self.action_space = spaces.Discrete(3)
            self.angle_range = 1/8
            # 定义离散动作映射表
            angle_options = [-1/8*np.pi, 0.0, 1/8*np.pi]
            self.action_idx_to_delta = angle_options
        
        elif mode == "horizon_discrete_adjust_5":
            self.action_space = spaces.Discrete(5)
            # 定义离散动作映射表
            angle_options = [-1/4*np.pi, -1/8*np.pi, 0.0, 1/8*np.pi, 1/4*np.pi]
            self.action_idx_to_delta = angle_options
        
        elif mode == "horizon_discrete_adjust_7":
            self.action_space = spaces.Discrete(7)
            # 定义离散动作映射表
            angle_options = [-3/8*np.pi, -1/4*np.pi, -1/8*np.pi, 0.0, 1/8*np.pi, 1/4*np.pi, 3/8*np.pi]
            self.action_idx_to_delta = angle_options

        else:
            raise ValueError(f"Unsupported control mode: '{mode}'")

    def _init_obs_space(self):
        if "dim" in self.env_params.get("observation", {}):
            dim = self.env_params["observation"]["dim"]
            self.observation_space = spaces.Box(low=0, high=1, shape=(dim,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)

    def _init_reward(self):
        reward_params = self.env_params["reward"]
        self.reward_components = []
        active = reward_params["active_components"]  # dict: {name: weight}

        if "target_progress_reward" in active:
            self.reward_components.append(TargetProgressReward("target_progress", active["target_progress_reward"]))

        if "nonlinear_collision_penalty" in active:
            self.reward_components.append(NonlinearCollisionPenalty("nonlinear_collision_penalty", active["nonlinear_collision_penalty"]))

        if "linear_collision_penalty" in active:
            self.reward_components.append(LinearCollisionPenalty("linear_collision_penalty", active["linear_collision_penalty"]))

        if "distance_to_obstacle_reward" in active:
            self.reward_components.append(DistanceToObstacleReward("distance_to_obstacle_reward", active["distance_to_obstacle_reward"]))

        if "direction_reward" in active:
            self.reward_components.append(DirectionReward("direction_reward", active["direction_reward"]))

        if "spherical_direction_reward" in active:
            self.reward_components.append(SphericalDirectionReward("spherical_direction_reward", active["spherical_direction_reward"]))

        if "nonlinear_spherical_direction_reward" in active:
            self.reward_components.append(NonlinearSphericalDirectionReward("nonlinear_spherical_direction_reward", active["nonlinear_spherical_direction_reward"]))

        if "velocity_reward" in active:
            self.reward_components.append(VelocityReward("velocity_reward", active["velocity_reward"]))

        if "image_nonlinear_collision_penalty" in active:
            self.reward_components.append(
                ImageNonlinearCollisionPenalty("image_nonlinear_collision_penalty", active["image_nonlinear_collision_penalty"])
            )
        
        if "image_nonlinear_collision_penalty_2" in active:
            self.reward_components.append(
                ImageNonlinearCollisionPenalty2("image_nonlinear_collision_penalty_2", active["image_nonlinear_collision_penalty_2"])
            )

        if "image_linear_collision_penalty" in active:
            self.reward_components.append(
                ImageLinearCollisionPenalty("image_linear_collision_penalty", active["image_linear_collision_penalty"])
            )

        if "cosine_spherical_direction_reward" in active:
            self.reward_components.append(
                CosineSphericalDirectionReward("cosine_spherical_direction_reward", active["cosine_spherical_direction_reward"])
            )
        
        if "cosine_spherical_direction_reward_2" in active:
            self.reward_components.append(
                CosineSphericalDirectionReward2("cosine_spherical_direction_reward_2", active["cosine_spherical_direction_reward_2"])
            )

        if "tanh_spherical_direction_reward" in active:
            self.reward_components.append(
                TanhSphericalDirectionReward("tanh_spherical_direction_reward", active["tanh_spherical_direction_reward"])
            )
        
        if "interpolation_spherical_direction_reward" in active:
            self.reward_components.append(
                InterpolationSphericalDirectionReward("interpolation_spherical_direction_reward", active["interpolation_spherical_direction_reward"])
            )

        if "terminal_reward" in active:
            arrival_reward = reward_params["extra_rewards"]["arrival_reward"]
            collision_penalty = reward_params["extra_rewards"]["collision_penalty"]
            self.reward_components.append(TerminalReward("terminal_reward", active["terminal_reward"], arrival_reward, collision_penalty))

    def reset(self):
        # 1. 重置仿真环境
        logging.info("仿真环境重置 ...")
        self.sim.reset()
        # 2. 重置计数器
        self.step_count = 0
        # 3. 重置初始位置、目标位置
        self.sim.drone.target_position = self.generate_target_positions()
        self.sim.drone.set_orientation()
        # 3. 获取初始观测
        obs = self.get_obs()
        # 初始化每个组件的回合累计奖励
        self.episode_total_reward = 0
        self.episode_component_rewards = {comp.name: 0.0 for comp in self.reward_components}
        return obs
    
    def step(self, action:np.ndarray):
        self.step_count += 1

        # === 1. 施加动作并推进仿真 ===
        velocity = self.compute_velocity_from_action(action)
        is_collided, nearest_info = self.sim.step(velocity, self._action_repeat)
        is_arrived = self.check_arrived()
        is_timeout = self.step_count >= self._max_episode_steps
        
        # === 2. 状态判断 ===
        done = is_collided or is_arrived or is_timeout

        self.sim.drone.set_orientation()

        # === 3. 观测 ===
        obs = self.get_obs()

       # === 4. 奖励计算（基于奖励组件系统）===
        total_reward, component_rewards = self.get_reward(obs, is_arrived, is_collided)  # get_reward返回总奖励 + 子项奖励字典

        # === 6. 附加info返回（包括细粒度奖励统计） ===
        info = dict(
            step_count=self.step_count,
            done=done,
            collision=is_collided,
            arrival=is_arrived,
            timeout=is_timeout,
        )

        for name, reward in component_rewards.items():
            self.episode_component_rewards[name] += reward  # 累加每个 reward component
        self.episode_total_reward += sum(component_rewards.values())  # 总奖励累计

        if done:
            # 将 episode 累积奖励加入 info（这样外部可以访问到它）
            info["episode/total_reward"] = self.episode_total_reward
            for name, total in self.episode_component_rewards.items():
                info[f"episode/{name}"] = total

        return obs, total_reward, done, info    

    def generate_target_positions(self):
        while True:
            # 从 scene_region 中采样位置
            x = np.random.uniform(self.sim.scene_region["x_min"], self.sim.scene_region["x_max"])
            y = np.random.uniform(self.sim.scene_region["y_min"], self.sim.scene_region["y_max"])
            z = np.random.uniform(self.sim.scene_region["z_min"], self.sim.scene_region["z_max"])
            target_position = [x, y, z]

            # Get the nearest obstacle distance for the target position
            is_collided, _ = self.check_target_collision(target_position, max_check_distance=10.0)
            
            if not is_collided:
                logging.info("🚁 目标位置安全，无碰撞")
                return target_position  # 如果没有碰撞，返回当前生成的位置
            else:
                logging.warning("🚨 目标位置与障碍物发生碰撞，重新生成位置")    
            
    def get_obs(self):
        """
        获取当前无人机的动态观测，根据需求选择观测特征。
        
        返回：
            np.array: 拼接后的观测数据
        """

        # 获取深度图信息（前方障碍物距离）每个像素是一个浮点数，介于 [0,1] 之间
        # 靠近相机的物体 → 深度值接近0
        # 远离相机的物体 → 深度值接近1
        # 如果看向空无一物的地方，深度值趋近于 far
        # 是二维矩阵，比如 shape = (240, 320)
        depth_image = self.sim.drone.get_depth_image()
        if "grid_shape" in self.env_params.get("observation", {}):
            grid_shape = self.env_params["observation"]["grid_shape"]
            grid_shape_tuple = ast.literal_eval(grid_shape)
        else:
            grid_shape_tuple = (4,4)
        obs = self.pool_depth_image(depth_image, grid_shape_tuple)
        flatten_obs = obs.flatten()

        # 拼接当前无人机的状态信息和深度图信息

        return flatten_obs
    
    def get_reward(self, obs, is_arrived, is_collided):
        """
        计算当前无人机的奖励值，组件化管理
        返回:
            total_reward: 综合奖励
            component_rewards: 每个子奖励项
        """
        total_reward = 0.0
        component_rewards = {}

        for component in self.reward_components:
            reward = component.compute(self, obs=obs, is_arrived=is_arrived, is_collided=is_collided)
            weighted_reward = reward * component.weight
            total_reward += weighted_reward
            component_rewards[component.name] = weighted_reward

        return total_reward, component_rewards

    def check_arrived(self,arrival_threshold=5.0):
        """
        检查是否到达目标点附近。

        参数：
            current_position: 当前无人机的位置 (x, y, z)
            target_position: 目标位置 (x, y, z)
            arrival_threshold: 到达目标的距离阈值
        
        返回：
            bool: 如果到达目标附近，返回 True；否则返回 False
        """
        distance_to_target = np.linalg.norm(np.array(self.sim.drone.state.position) - np.array(self.sim.drone.target_position))
        return distance_to_target <= arrival_threshold  # 如果距离小于阈值，认为到达目标

    def compute_velocity_from_action(self, action: np.ndarray):
        """
        根据指定 mode 解释动作，并执行对应控制。

        参数:
            action (np.ndarray): 动作向量
            mode (str): 控制模式，可为 'cartesian', 'spherical', 'adjust'
        """
        mode = self.env_params["action"]["type"]
        
        if mode == "cartesian":
            new_velocity = np.array(action, dtype=np.float32)

        elif mode == "spherical":
            # 绝对球坐标 → 笛卡尔
            v, theta, phi = action
            vx = v * np.sin(theta) * np.cos(phi)
            vy = v * np.sin(theta) * np.sin(phi)
            vz = v * np.cos(theta)
            new_velocity = np.array([vx, vy, vz], dtype=np.float32)

        elif mode == "adjust":
            delta_theta = action[0]
            delta_phi   = action[1]

            # 目标速度设定
            v_horiz = 15.0  # 水平速度
            v_vert = 5.0    # 垂直速度
            # 获取当前位置与目标位置
            current_position = np.array(self.sim.drone.state.position)
            target_position = np.array(self.sim.drone.target_position)

            # 用目标方向替代当前速度方向
            direction_vector = target_position - current_position
            norm = np.linalg.norm(direction_vector)

            if norm < 1e-3:
                theta = np.pi / 2
                phi = 0.0
            else:
                # 计算从当前位置指向目标位置的方向角
                theta = np.arccos(direction_vector[2] / norm)  # 极角（俯仰）
                phi = np.arctan2(direction_vector[1], direction_vector[0])  # 方位角（偏航）

            theta_new = np.clip(theta + delta_theta, 0, np.pi)
            phi_new = phi + delta_phi

            # 构造单位方向向量（方向确定，但模长与速度无关）
            vx_unit = np.sin(theta_new) * np.cos(phi_new)
            vy_unit = np.sin(theta_new) * np.sin(phi_new)
            vz_unit = np.cos(theta_new)

            # 归一化水平分量向量
            horiz_norm = np.linalg.norm([vx_unit, vy_unit])
            if horiz_norm < 1e-6:
                vx = 0.0
                vy = 0.0
            else:
                vx = v_horiz * (vx_unit / horiz_norm)
                vy = v_horiz * (vy_unit / horiz_norm)

            # 垂直速度直接设为固定模长（方向由 theta_new 决定）
            vz = v_vert * np.sign(vz_unit)

            new_velocity = np.array([vx, vy, vz], dtype=np.float32)
        
        elif mode == "discrete_adjust":
            delta_theta, delta_phi = self.action_idx_to_delta[action]
            
            # # 目标速度设定
            speed = 15.0
            # 获取当前位置与目标位置
            current_position = np.array(self.sim.drone.state.position)
            target_position = np.array(self.sim.drone.target_position)

            # 用目标方向替代当前速度方向
            direction_vector = target_position - current_position
            norm = np.linalg.norm(direction_vector)

            if norm < 1e-3:
                theta = np.pi / 2
                phi = 0.0
            else:
                # 计算从当前位置指向目标位置的方向角
                theta = np.arccos(direction_vector[2] / norm)  # 极角（俯仰）
                phi = np.arctan2(direction_vector[1], direction_vector[0])  # 方位角（偏航）

            theta_new = np.clip(theta + delta_theta, 0, np.pi)
            phi_new = phi + delta_phi

            # 构造单位方向向量（方向确定，但模长与速度无关）
            vx_unit = np.sin(theta_new) * np.cos(phi_new)
            vy_unit = np.sin(theta_new) * np.sin(phi_new)
            vz_unit = np.cos(theta_new)

            # 计算速度向量（单位向量乘以速度大小）
            vx = speed * vx_unit
            vy = speed * vy_unit
            vz = speed * vz_unit

            new_velocity = np.array([vx, vy, vz], dtype=np.float32)
        
        elif mode in ["discrete_adjust_2", "horizon_discrete_adjust_2"]:
            # 参数
            v_horiz = 15.0  # 水平速度基准
            v_vert_max = 5.0  # 垂直速度上限
            horizontal_threshold = 2.0  # 水平距离小于该值认为已对准目标水平位置

            delta_theta, delta_phi = self.action_idx_to_delta[action]
            # 获取当前和目标位置
            current_position = np.array(self.sim.drone.state.position)
            target_position = np.array(self.sim.drone.target_position)
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
                    scale = abs(vz_target / dir_unit[2])  # 以 vz 固定为目标，缩放整向量
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

                vz = vz_unit * v_horiz  # 初始 vz，等比例

                # 限制 vz 不超过最大值
                if abs(vz) > v_vert_max:
                    vz = v_vert_max * np.sign(vz)

                new_velocity = np.array([vx, vy, vz], dtype=np.float32)

        elif mode in ["horizon_discrete_adjust_3", "horizon_discrete_adjust_5", "horizon_discrete_adjust_7"]:
            delta_phi = self.action_idx_to_delta[action]

            # 目标速度设定
            speed = 15.0
            # 获取当前位置与目标位置
            current_position = np.array(self.sim.drone.state.position)
            target_position = np.array(self.sim.drone.target_position)

            # 用目标方向替代当前速度方向
            direction_vector = target_position - current_position
            norm = np.linalg.norm(direction_vector)

            if norm < 1e-3:
                theta = np.pi / 2
                phi = 0.0
            else:
                # 计算从当前位置指向目标位置的方向角
                theta = np.arccos(direction_vector[2] / norm)  # 极角（俯仰）
                phi = np.arctan2(direction_vector[1], direction_vector[0])  # 方位角（偏航）

            theta_new = theta
            phi_new = phi + delta_phi

            # 构造单位方向向量（方向确定，但模长与速度无关）
            vx_unit = np.sin(theta_new) * np.cos(phi_new)
            vy_unit = np.sin(theta_new) * np.sin(phi_new)
            vz_unit = np.cos(theta_new)

            # 计算速度向量（单位向量乘以速度大小）
            vx = speed * vx_unit
            vy = speed * vy_unit
            vz = speed * vz_unit

            new_velocity = np.array([vx, vy, vz], dtype=np.float32)

        else:
            raise ValueError(f"Unsupported action mode: '{mode}'. Expected 'cartesian', 'spherical', or 'adjust'.")
             
        return new_velocity

    def pool_depth_image(self, depth_image, grid_shape=(4, 4)):
        """
        对深度图进行最小池化，按网格划分。
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

    def check_target_collision(self, target_position, max_check_distance = 10.0):
        target_radius = 1
        # 创建可视化形状（红色半透明球）
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=target_radius,
            rgbaColor=[1, 0, 0, 0.5],  # 红色，半透明
        )

        # 创建碰撞形状
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=target_radius
        )

        # 创建带可视化和碰撞的临时球体
        target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=target_position
        )
        min_distance = max_check_distance  # Initialize with max_check_distance
        nearest_info = None

        # Iterate through all bodies, excluding the current object (target_id)
        for body_id in range(p.getNumBodies()):
            if body_id != target_id:
                # Retrieve closest points between self and another body
                closest_points = p.getClosestPoints(
                    bodyA=target_id,
                    bodyB=body_id,
                    distance=max_check_distance
                )

                for point in closest_points:
                    # Point[8] contains the distance between two bodies
                    distance = point[8]
                    if distance < min_distance:
                        min_distance = distance
                        try:
                            # Attempt to retrieve the body name, handle any exceptions
                            body_name = p.getBodyInfo(body_id)[1].decode('utf-8')
                        except Exception as e:
                            body_name = f"Unknown (ID: {body_id})"
                        
                        nearest_info = {
                            "id": body_id,
                            "name": body_name,
                            "distance": distance,
                            "position": point[6]  # Point[6] contains the contact position
                        }
        p.removeBody(target_id)  # 清除临时球体
        if nearest_info:
            # p.removeBody(target_id)  # 清除临时球体
            return True, nearest_info
        else:
            return False, None

