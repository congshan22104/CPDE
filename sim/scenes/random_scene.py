import pybullet as p
import pybullet_data
import numpy as np
import random
import os

class RandomScene:
    def __init__(self, scene_size_x=100.0, scene_size_y=100.0, scene_size_z=50.0,
                 num_obstacles=50, min_radius=1.0, max_radius=3.0,
                 min_height=2.0, max_height=6.0):
        self.scene_size_x = scene_size_x  # ✅ 地图长
        self.scene_size_y = scene_size_y  # ✅ 地图宽
        self.scene_size_z = scene_size_z  # ✅ 地图高（用于体素）
        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_height = min_height
        self.max_height = max_height

        self.obstacle_list = []
        self.voxel_map = None

    def build(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", globalScaling=60)
        self._generate_obstacles()

    def _generate_obstacles(self):
        self.obstacle_list.clear()
        x = 0
        y = 0
        for _ in range(self.num_obstacles):
            is_cylinder = random.choice([True, False])
            color = [random.random(), random.random(), random.random(), 1.0]

            if is_cylinder:
                radius = random.uniform(self.min_radius, self.max_radius)
                height = random.uniform(self.min_height, self.max_height)
                col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
                vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
                length = width = radius * 2
            else:
                length = random.uniform(self.min_radius, self.max_radius)
                width = random.uniform(self.min_radius, self.max_radius)
                height = random.uniform(self.min_height, self.max_height)
                col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2])
                vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, width/2, height/2], rgbaColor=color)

            z = height / 2
            p.createMultiBody(0, col_shape, vis_shape, basePosition=[x, y, z])

            self.obstacle_list.append({
                'x': x, 'y': y, 'z': z, # 障碍物的中心点
                'is_cylinder': is_cylinder,
                'radius': radius if is_cylinder else None,
                'length': length if not is_cylinder else None,
                'width': width if not is_cylinder else None,
                'height': height,
                'color': color
            })

            x = random.uniform(-self.scene_size_x / 2, self.scene_size_x / 2)  # ✅ 长轴范围
            y = random.uniform(-self.scene_size_y / 2, self.scene_size_y / 2)  # ✅ 宽轴范围

    def get_obstacles(self):
        return self.obstacle_list
