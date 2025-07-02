import pybullet as p
import pybullet_data
import logging
import os
import numpy as np 
import time

from sim.agents import DroneAgent
from sim.scenes import RandomScene, VoxelizedRandomScene, RealScene

class World:
    def __init__(self, use_gui, scene_type, scene_region, obstacle_params, drone_params, voxel_size=None, building_path=""):
        self.use_gui = use_gui

        # 参数缓存
        self.scene_type = scene_type
        self.scene_region = scene_region
        self.obstacle_params = obstacle_params
        self.drone_params = drone_params
        self.voxel_size = voxel_size
        self.building_path = building_path

        # 初始化内容
        self.scene = None
        self.drone = None

        self.reset()

    def _connect_pybullet(self):
        if p.getConnectionInfo()['isConnected']:
            logging.info("已连接到 PyBullet，正在断开以避免重复连接。")
            p.disconnect()
        if  self.use_gui:
            p.connect(p.GUI)
            self._setup_camera()
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

    def _setup_camera(self, camera_target=[0, 0, 0], camera_yaw=45, camera_pitch=-45):
        p.resetDebugVisualizerCamera(
            cameraDistance=600,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target
        )

    def _load_ground(self):
        p.loadURDF("plane.urdf")

    def _build_scene(self):
        logging.info("🔧 Building scene ...")
        # 场景尺寸
        scene_size_x = self.scene_region["x_max"] - self.scene_region["x_min"]
        scene_size_y = self.scene_region["y_max"] - self.scene_region["y_min"]
        scene_size_z = self.scene_region["z_max"] - self.scene_region["z_min"]
        if self.scene_type == "random":
            self.scene = RandomScene(
                scene_size_x=scene_size_x,
                scene_size_y=scene_size_y,
                scene_size_z=scene_size_z,
                num_obstacles=self.obstacle_params["num_obstacles"],
                min_radius=self.obstacle_params["min_radius"],
                max_radius=self.obstacle_params["max_radius"],
                min_height=self.obstacle_params["min_height"],
                max_height=self.obstacle_params["max_height"]
            )
        elif self.scene_type == "real":
            self.scene = RealScene(
                scene_size_x=scene_size_x,
                scene_size_y=scene_size_y,
                scene_size_z=scene_size_z,
                building_path=self.building_path
            )
        elif self.scene_type == "voxelized":
            self.scene = VoxelizedRandomScene(
                scene_size_x=scene_size_x,
                scene_size_y=scene_size_y,
                scene_size_z=scene_size_z,
                num_obstacles=self.obstacle_params["num_obstacles"],
                min_radius=self.obstacle_params["min_radius"],
                max_radius=self.obstacle_params["max_radius"],
                min_height=self.obstacle_params["min_height"],
                max_height=self.obstacle_params["max_height"],
                voxel_size=self.voxel_size
            )
        else:
            raise ValueError(f"Unsupported scene_type: {self.scene_type}")

        self.scene.build()

    def _spawn_drones(self):
        """
        初始化红队和蓝队的无人机位置和目标位置。
        如果提供了初始位置和目标位置，则使用提供的数据；
        否则，自动生成安全的位置。
        """
        # 初始化蓝队无人机列表
        self.chasers = []
        chaser_inits   = self.drone_params.get('init_positions', {}).get('chaser', [])
        chaser_targets = self.drone_params.get('target_positions', {}).get('chaser', [])
        colos = [[0, 0, 1, 1],[0, 1, 0, 1],[0, 1, 1, 1]]
        # 初始化（chaser）无人机
        for i in range(3):
            init_pos   = chaser_inits[i]
            target_pos = chaser_targets[i]

            drone = self._initialize_single_drone(
                team_name='chaser',
                init_position=init_pos,
                target_position=target_pos,
                min_safe_distance=self.drone_params.get('min_safe_distance', 10.0),
                urdf_path=self.drone_params.get('urdf_path'),
                color=colos[i]  # Blue color for chasers
            )
            self.chasers.append(drone)

        # 初始化红队无人机（runner）
        self.runners = []
        runner_inits   = self.drone_params.get('init_positions', {}).get('runner', [])
        runner_targets = self.drone_params.get('target_positions', {}).get('runner', [])

        # 初始化（runner）无人机
        for i in range(1):
            init_pos   = runner_inits[i]
            target_pos = runner_targets[i]

            drone = self._initialize_single_drone(
                team_name='runner',
                init_position=init_pos,
                target_position=target_pos,
                min_safe_distance=self.drone_params.get('min_safe_distance', 10.0),
                urdf_path=self.drone_params.get('urdf_path'),
                color=[1, 0, 0, 1]  # Red color for runners
            )
            self.runners.append(drone)

        # Log the success of drone initialization
        logging.info("所有无人机初始化完成！")

    def _initialize_single_drone(self, 
                                team_name, 
                                color,
                                init_position=None, 
                                target_position=None, 
                                min_safe_distance=10.0, 
                                urdf_path="assets/cf2x.urdf"):
        """
        初始化单个无人机。
        """
        # 初始位置
        if init_position:
            init_pos = init_position
            logging.info(f"🚁 使用提供的 {team_name} 队初始位置: {init_pos}")
        else:
            init_pos = self._generate_safe_position(min_safe_distance)
            logging.info(f"🚁 自动生成的 {team_name} 队初始位置: {init_pos}")

        # 目标位置
        if target_position:
            target_pos = target_position
            logging.info(f"🎯 使用提供的 {team_name} 队目标位置: {target_pos}")
        else:
            target_pos = self._generate_safe_position(min_safe_distance)
            logging.info(f"🎯 自动生成的 {team_name} 队目标位置: {target_pos}")

        # 创建单个无人机实例
        drone = DroneAgent(
            index=0,  # Since we are initializing just one drone
            team=team_name,
            init_pos=init_pos,
            target_pos=target_pos,
            urdf_path=urdf_path,
            color=color,
        )

        logging.info(f"✅ {team_name} 队单个无人机初始化完成")

        return drone

    def reset(self):
        logging.info("重置仿真环境...")
        self._connect_pybullet()
        self._load_ground()
        self._build_scene()
        self._spawn_drones()
        logging.info("仿真环境重置完成。")

    def step(self, chaser_velocities, runner_velocities, num_repeats=30):
        """
        对红方无人机进行碰撞检测，并在发生碰撞时标记为死亡。

        参数：
            chaser_velocities (list): 蓝队无人机的速度列表。
            runner_velocities (list): 红队无人机的速度列表。
            num_repeats (int): 每次动作重复的次数。
            collision_threshold (float): 碰撞检测的距离阈值。
            collision_check_interval (int): 碰撞检测的间隔步数。
        """
        p.setRealTimeSimulation(0)  # 关闭实时模拟
        p.setTimeStep(1./240.)  # 设置时间步
        for i in range(num_repeats):
            # 应用速度控制 如果不重新设置 这个速度会减慢
            for drone, vel in zip(self.chasers, chaser_velocities):
                p.resetBaseVelocity(drone.id, linearVelocity=vel)
            for drone, vel in zip(self.runners, runner_velocities):
                p.resetBaseVelocity(drone.id, linearVelocity=vel)
            p.stepSimulation()
        # 更新状态和绘制轨迹
        for drone in self.chasers + self.runners:
            drone.update_state()
            if self.use_gui:
                drone.draw_trajectory()

    def _generate_safe_position(self, min_safe_distance=10.0):
        """
        生成指定目标位置，确保每个位置与障碍物不发生碰撞。
        """
        while True:
            x = np.random.uniform(self.scene_region["x_min"], self.scene_region["x_max"])
            y = np.random.uniform(self.scene_region["y_min"], self.scene_region["y_max"])
            z = np.random.uniform(self.scene_region["z_min"], self.scene_region["z_max"])
            position = [x, y, z]

            distance_to_nearest_obstacle = self.compute_point_to_nearest_obstacle_distance(
                position, max_check_distance=10.0)

            if distance_to_nearest_obstacle > min_safe_distance:
                logging.info(f"🎯 位置安全: {position}")
                return position
            else:
                logging.warning("🚨 位置与障碍物发生碰撞，重新生成位置")

    def compute_point_to_nearest_obstacle_distance(self, point, max_check_distance=10.0):
        """
        计算给定点到最近障碍物的距离。

        参数：
            point (list or np.ndarray): 3D 坐标 [x, y, z]
            max_check_distance (float): 最大检测范围（射线长度）

        返回：
            float: 到最近障碍物的距离。如果未命中，返回 max_check_distance。
        """
        target_radius = 0.01  # 可根据需要调整半径大小

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
            basePosition=point
        )

        min_distance = max_check_distance  # 初始化为最大检测距离

        # 遍历所有物体，排除当前临时球体
        for body_id in range(p.getNumBodies()):
            if body_id != target_id:
                # 获取当前物体与其他物体之间的最近点信息
                closest_points = p.getClosestPoints(
                    bodyA=target_id,
                    bodyB=body_id,
                    distance=max_check_distance
                )

                for pt in closest_points:
                    distance = pt[8]  # 第9个元素是距离信息
                    if distance < min_distance:
                        min_distance = distance

        # 移除临时球体
        p.removeBody(target_id)

        return min_distance
