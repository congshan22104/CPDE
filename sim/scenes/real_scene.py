
import pybullet as p
import logging

class RealScene:
    def __init__(self, mesh_path, scale=1.0, position=[0, 0, 0], orientation=[0, 0, 0, 1]):
        """
        加载三维建筑地图模型（如.obj或.urdf）

        参数:
            mesh_path (str): 模型文件路径（.obj/.stl/.dae 等）
            scale (float): 缩放比例
            position (list): [x, y, z] 模型位置
            orientation (list): 四元数 [x, y, z, w]
        """
        self.mesh_path = mesh_path
        self.scale = scale
        self.position = position
        self.orientation = orientation
        self.map_id = None

    def build(self):
        logging.info("🗺️ 加载地图模型: %s", self.mesh_path)
        try:
            self.map_visual_shape = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=self.mesh_path,
                meshScale=[self.scale] * 3,
                rgbaColor=[0.8, 0.8, 0.8, 1.0],  # Light gray color with transparency,
            )

            self.map_collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=self.mesh_path,
                meshScale=[self.scale] * 3,
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
            )

            self.map_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=self.map_collision_shape,
                baseVisualShapeIndex=self.map_visual_shape,
                basePosition=self.position,
                baseOrientation=self.orientation,
            )

            p.changeDynamics(self.map_id, -1, lateralFriction=0.8, restitution=0.5)
            logging.info("✅ 地图加载完成")

        except Exception as e:
            logging.error("❌ 地图加载失败: %s", e)
            raise
