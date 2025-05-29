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
        while True:
            init_pos_config = self.drone_params["init_pos"]

            if init_pos_config == "random":
                # 从 scene_region 中采样位置
                x = np.random.uniform(self.scene_region["x_min"], self.scene_region["x_max"])
                y = np.random.uniform(self.scene_region["y_min"], self.scene_region["y_max"])
                z = np.random.uniform(self.scene_region["z_min"], self.scene_region["z_max"])
                init_pos = [x, y, z]
                logging.info(f"🚁 随机生成无人机位置: {init_pos}")
            else:
                init_pos = init_pos_config
                logging.info(f"🚁 固定无人机位置: {init_pos}")

            urdf_path = self.drone_params["urdf_path"]
            self.drone = DroneAgent(
                index=0,
                team="blue",
                init_pos=init_pos,
                target_pos=[0,1,1],
                urdf_path=urdf_path,
                color=[0, 0, 1, 1],
            )
            
            # 检查位置是否与障碍物碰撞
            is_collided, _ = self.drone.check_collision(threshold=5.0)
            if not is_collided:
                logging.info("🚁 初始位置安全，无碰撞")
                return init_pos  # 如果没有碰撞，返回当前生成的位置
            else:
                logging.warning("🚨 初始位置与障碍物发生碰撞，重新生成位置")
                # 删除当前不合适的无人机，避免占用资源
                self.drone.remove()  # 假设你有一个remove方法卸载模型，否则需要写卸载代码

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