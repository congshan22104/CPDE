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

        # 场景尺寸
        self.scene_size_x = scene_region["x_max"] - scene_region["x_min"]
        self.scene_size_y = scene_region["y_max"] - scene_region["y_min"]
        self.scene_size_z = scene_region["z_max"] - scene_region["z_min"]

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
        if self.use_gui:
            p.connect(p.GUI)
            self._setup_camera()
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

    def _setup_camera(self):
        camera_target = [0, 0, 0]
        camera_yaw = 45
        camera_pitch = -45
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

        if self.scene_type == "random":
            self.scene = RandomScene(
                scene_size_x=self.scene_size_x,
                scene_size_y=self.scene_size_y,
                scene_size_z=self.scene_size_z,
                num_obstacles=self.obstacle_params["num_obstacles"],
                min_radius=self.obstacle_params["min_radius"],
                max_radius=self.obstacle_params["max_radius"],
                min_height=self.obstacle_params["min_height"],
                max_height=self.obstacle_params["max_height"]
            )
        elif self.scene_type == "real":
            self.scene = RealScene(
                scene_size_x=self.scene_size_x,
                scene_size_y=self.scene_size_y,
                scene_size_z=self.scene_size_z,
                building_path=self.building_path
            )
        elif self.scene_type == "voxelized":
            self.scene = VoxelizedRandomScene(
                scene_size_x=self.scene_size_x,
                scene_size_y=self.scene_size_y,
                scene_size_z=self.scene_size_z,
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

    def _spawn_drone(self):
        drone_init   = self.drone_params.get('init_position', None)
        drone_target = self.drone_params.get('target_position', None)
        
        self.drone = self._initialize_single_drone(
            team_name='chaser',
            init_position=drone_init,
            target_position=drone_target,
            min_safe_distance=self.drone_params.get('min_safe_distance', 10.0),
            urdf_path=self.drone_params.get('urdf_path'),
            color=[0, 0, 1, 1]  # Blue color for chasers
        )
             
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

            if distance_to_nearest_obstacle >= min_safe_distance:
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

        # 创建碰撞形状
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=target_radius
        )

        # 创建带可视化和碰撞的临时球体
        target_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=-1,
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
    
    def reset(self):
        logging.info("重置仿真环境...")
        self._connect_pybullet()
        self._load_ground()
        self._build_scene()
        self._spawn_drone()
        logging.info("仿真环境重置完成。")

    def step(self, velocity, num_steps=30):
        is_collided = False
        collision_check_interval = 30
        for i in range(num_steps):
            p.resetBaseVelocity(self.drone.id, linearVelocity=velocity)
            p.stepSimulation()
            # time.sleep(1. / 240.)
            if i % collision_check_interval == 0:
                is_collided, nearest_info = self.drone.check_collision()
                if is_collided:
                    break
        if self.use_gui:
            self.drone.draw_trajectory()
        self.drone.update_state()
        
        return is_collided, nearest_info
