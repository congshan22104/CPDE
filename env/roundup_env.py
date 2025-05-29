import gym
from gym import spaces
import numpy as np
import random
import logging
import os
import torch
import yaml
import ast
import argparse
from model.ppo_discrete_rnn.ppo_discrete_rnn import PPO_discrete_RNN
from sim.roundup_world import World
from env.wrappers.ma_reward_wrapper import DistanceReward, AngleReward


class ThreeChaserOneRunnerEnv(gym.Env):
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
        self.num_chaser = self.env_params['drone']['num_chaser']
        self.num_runner = self.env_params['drone']['num_runner']
                
        # Initialize simulation, action, observation spaces, and rewards
        self._init_simulation()

        self.agents = {f"agent_{i}": self.sim.chasers[i] for i in range(self.num_chaser)}
        self._init_action_spaces()
        self._init_obs_spaces()
        self._init_reward()

        # Load PPO model
        self._load_navigation_policy()

        # Initialize tracking for episode rewards
        self.episode_total_reward = 0
        self.episode_component_rewards = {comp.name: 0.0 for comp in self.reward_components}

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
        if mode == 'discrete':
            from gymnasium.spaces import Discrete
            self.action_spaces = {agent_id: Discrete(6) for agent_id in self.agents.keys()}  # 6 discrete actions
        elif mode == 'continuous':
            from gymnasium.spaces import Box
            low = np.array([0.0, 0.0, -np.pi], dtype=np.float32)
            high = np.array([15.0, 5.0, np.pi], dtype=np.float32)
            self.action_spaces = {agent_id: Box(low, high, dtype=np.float32) for agent_id in self.agents.keys()}

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

        self.observation_spaces = {
            agent: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents.keys()
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

    def _load_navigation_policy(self):
        """
        Loads the PPO model and its weights.
        """
        folder_path = "/home/congshan/uav/multi_uav_navigation/navigation_policy"
        ppo_config_path = os.path.join(folder_path, "config", "ppo_config.yaml")
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

    def reset(self):
        """
        Resets the environment for a new episode and returns the initial observation.
        """
        logging.info("Resetting simulation environment ...")
        self.sim.reset()
        self.step_count = 0
        # self.sim.drones.target_position = self.generate_target_positions()
        # self.sim.set_orientation_drones()

        # Initialize rewards
        self.episode_total_reward = 0
        self.episode_component_rewards = {comp.name: 0.0 for comp in self.reward_components}

        obs = self.get_obs()

        return obs

    def step(self, actions):
        """
        Advances the simulation by one step based on the actions taken by the agents.
        """
        self.step_count += 1
        max_episode_steps = self.env_params['episode']['max_episode_timesteps']
        runner_velocity = [random.uniform(-5.0, 5.0) for _ in range(3)]  # Random runner velocity

        is_collided = False
        chaser_velocities = []

        # Initialize per-agent status for terminated, truncated, and collided
        agents_terminated = {idx: False for idx in self.agents.keys()}
        agents_truncated = {idx: False for idx in self.agents.keys()}
        agents_collided = {idx: False for idx in self.agents.keys()}
        agents_captured = {idx: False for idx in self.agents.keys()}

        for _ in range(8):  # Number of repeats
            for idx, chaser in self.agents.items():
                # Check if the drone has collided
                if chaser.state.collided:
                    # If collided, set the velocity to [0, 0, 0]
                    velocity = np.array([0.0, 0.0, 0.0])
                else:
                    # Otherwise, calculate the velocity based on the action
                    chaser.target_position = actions[idx]
                    local_obs = self.get_local_obs(chaser)
                    nav_action, _ = self.navigation_policy.choose_action(local_obs, evaluate=True)
                    velocity = self.compute_velocity_from_action(chaser, nav_action)

                # Append the calculated velocity (or [0, 0, 0] if collided) to the list
                chaser_velocities.append(velocity)

            self.sim.step(chaser_velocities, runner_velocity, num_repeats=30)

            # Iterate over each drone in the simulation
            for idx, drone in self.agents.items():
                # Check if the drone has captured the runner
                captured = self.check_capture(drone, self.sim.runners[0])  # Assuming runner is the first one
                
                # Check if the drone has collided (you would need a method for this)
                collided = drone.state.collided
                
                # Check if the drone has reached the time limit
                reached_time_limit = self.step_count >= self.env_params['episode']['max_episode_timesteps']

                terminated = captured or collided or reached_time_limit
                
                # Store the results in the status dictionaries for this drone
                agents_captured[idx] = captured
                agents_collided[idx] = collided
                agents_truncated[idx] = reached_time_limit
                agents_terminated[idx] = terminated
            


        # Calculate reward
        total_reward, component_rewards = self.get_rewards()

        # Information to return
        info = {
            "total_reward": total_reward,
            "component_rewards": component_rewards,
            "terminated": agents_terminated,
            "captured": agents_captured,
            "truncated": agents_truncated,
            "collided": agents_collided,
        }

        obs = self.get_obs()
        return obs, total_reward, agents_terminated, agents_truncated, info


    def get_obs(self):
        """
        Returns the observation for each agent, which includes positions, velocities, and other relevant features.
        """
        features = self.env_params['observation']['features']
        # Initialize observations as a dictionary
        obs = {}
        
        # Access runner state
        red = self.sim.runners[0]
        red_pos, red_vel = red.state.position, red.state.linear_velocity

        for i, chaser in self.agents.items():
            obs_i = []  # Individual observation for the current chaser
            
            # Get current chaser position
            self_pos = chaser.state.position

            if "target_ralative" in features:
                rel = red_pos - self_pos
                r = np.linalg.norm(rel)
                theta = np.arccos(rel[2] / (r + 1e-8)) if r > 1e-8 else 0.0
                phi = np.arctan2(rel[1], rel[0]) if r > 1e-8 else 0.0
                obs_i.append(np.array([r, theta, phi]))

            if "runner_velocity" in features:
                rel_vel = red_vel
                v = np.linalg.norm(rel_vel)
                theta = np.arccos(rel_vel[2] / (v + 1e-8)) if v > 1e-8 else 0.0
                phi = np.arctan2(rel_vel[1], rel_vel[0]) if v > 1e-8 else 0.0
                obs_i.append(np.array([v, theta, phi]))

            if "teammate_ralative" in features:
                teammates = []
                for j, teammate in self.agents.items():
                    if j == i:  # Skip the current chaser
                        continue
                    rel = teammate.state.position - self_pos
                    r = np.linalg.norm(rel)
                    theta = np.arccos(rel[2] / (r + 1e-8)) if r > 1e-8 else 0.0
                    phi = np.arctan2(rel[1], rel[0]) if r > 1e-8 else 0.0
                    teammates.append(np.array([r, theta, phi]))
                if teammates:
                    obs_i.append(np.concatenate(teammates))

            # Check if the observation for this chaser is not empty
            if obs_i:
                obs[f"{i}"] = np.concatenate(obs_i)
            else:
                raise ValueError(f"[get_obs] obs_i is empty for drone {i}. Check enabled features.")

        return obs


    def get_rewards(self):
        """
        Computes the total reward for each drone and the component rewards.
        
        Returns:
        - total_rewards: A dictionary where the key is the drone's name, and the value is its total reward.
        - component_rewards: A dictionary where the key is the drone's name, and the value is another dictionary of component rewards for that drone.
        """
        total_rewards = {}
        component_rewards = {}

        for drone_name, chaser in self.agents.items():  # Loop over each drone
            total_reward = 0
            component_rewards_for_drone = {}

            # Get the states of the drone and runner
            chaser_state = chaser.state
            runner_state = self.sim.runners[0].state  # Assuming the runner is the first one

            for component in self.reward_components:  # Loop over reward components
                component_reward = component.compute(chaser_state, runner_state)
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
        获取当前无人机的动态观测，根据需求选择观测特征。
        
        返回：
            np.array: 拼接后的观测数据
        """

        # 获取深度图信息（前方障碍物距离）每个像素是一个浮点数，介于 [0,1] 之间
        # 靠近相机的物体 → 深度值接近0
        # 远离相机的物体 → 深度值接近1
        # 如果看向空无一物的地方，深度值趋近于 far
        # 是二维矩阵，比如 shape = (240, 320)
        depth_image = chaser.get_depth_image()
        if "grid_shape" in self.env_params.get("observation", {}):
            grid_shape = self.env_params["observation"]["grid_shape"]
            grid_shape_tuple = ast.literal_eval(grid_shape)
        else:
            grid_shape_tuple = (4,4)
        obs = self.pool_depth_image(depth_image, grid_shape_tuple)
        flatten_obs = obs.flatten()

        return flatten_obs

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

        return new_velocity