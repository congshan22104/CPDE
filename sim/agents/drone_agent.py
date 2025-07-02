import numpy as np
import logging
import pybullet as p
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import os
import matplotlib.pyplot as plt
import datetime

@dataclass
class DroneState:
    position: np.ndarray
    orientation: np.ndarray
    euler: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    min_distance_to_obstacle: np.ndarray
    collided: bool

class DroneAgent:
    def __init__(self, index, team, init_pos, target_pos, urdf_path, color):
        """
        初始化单架无人机智能体。
        """
        self.index = index
        self.team = team
        self.init_pos = init_pos
        self.color = color

        self._load_model(urdf_path)
        self._set_visual()
        self._set_dynamics()
        self.trajectory = []
        self.target_position = target_pos
        self.state = self.get_state()
        # self.start_marker_id = self._create_marker(self.init_pos, color=[0, 1, 0, 1])   # 半透明绿色球表示起点
        # self.end_marker_id = self._create_marker(self.target_position, color=[1, 0, 0, 1])  # 半透明红色球表示终点

        logging.info("[Init] %s #%d | ID=%d | Pos=%s", team.capitalize(), index, self.id, init_pos)
    
    def _create_marker(self, position, color, radius=10):
        """
        创建一个纯可视标记（无碰撞体积）。
        """
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            visualFramePosition=[0, 0, 0],
        )

        marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            useMaximalCoordinates=True
        )

        return marker_id
    
    def _load_model(self, urdf_path):
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.id = p.loadURDF(
            fileName=urdf_path,
            basePosition=self.init_pos,
            baseOrientation=orientation,
            globalScaling=1.0
        )

    def _set_visual(self):
        p.changeVisualShape(self.id, -1, rgbaColor=self.color)

    def _set_dynamics(self):
        p.changeDynamics(
            self.id,
            -1,
            restitution=0.0,
            lateralFriction=1.0,
            linearDamping=0.3,
            angularDamping=0.3
        )

    def apply_force(self, force):
        """
        对无人机施加外力，影响无人机的运动

        参数:
        - force: 3D 向量(np.ndarray 或 list)
        """
        force = force.squeeze().tolist()
        pos, _ = p.getBasePositionAndOrientation(self.id)
        try:
            p.applyExternalForce(
                objectUniqueId=self.id,
                linkIndex=-1,
                forceObj=force,
                posObj=pos,
                flags=p.WORLD_FRAME
            )
        except Exception as e:
            logging.error("施加外力失败 [ID=%d]: %s", self.id, e)
    
    def set_velocity(self, velocity):
        """
        设置无人机的速度，velocity 是一个长度为 3 的列表或数组，
        包含了 x, y, z 方向的速度分量
        """
        linear_velocity = velocity  # 传入的速度就是要设置的线性速度
        p.resetBaseVelocity(self.id, linearVelocity=linear_velocity)
    
    def set_orientation(self):
        # 1. 计算从当前位置指向目标位置的方向向量
        direction_vector = self.target_position - self.state.position

        # 2. 提取水平分量（忽略 z 轴/竖直方向）
        horizontal_direction = np.array([direction_vector[0], direction_vector[1], 0.0])
        horizontal_distance = np.linalg.norm(horizontal_direction)

        # 3. 如果水平速度太小，保持默认朝向（单位矩阵）
        if horizontal_distance < 1e-3:
            return np.eye(3)

        # 4. 设置 x 轴为水平方向
        x_axis = horizontal_direction / horizontal_distance

        # 5. 设置世界 z 轴为上方向
        world_up = np.array([0.0, 0.0, 1.0])

        # 6. 计算 y 轴为 world_up × x_axis
        y_axis = np.cross(world_up, x_axis)
        y_axis /= (np.linalg.norm(y_axis) + 1e-6)

        # 7. 计算 z 轴为 x_axis × y_axis（保证右手坐标系）
        z_axis = np.cross(x_axis, y_axis)

        # 8. 组装旋转矩阵
        rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])

        # 9. 转换为四元数
        orn = Rotation.from_matrix(rotation_matrix).as_quat()

        # 10. 更新朝向（保持位置不变）
        p.resetBasePositionAndOrientation(self.id, self.state.position, orn)

        
        return orn

    def draw_trajectory(self, width=10, duration=0):
        """
        更新无人机轨迹并在 PyBullet 中绘制轨迹线段。

        参数:
            width (float): 轨迹线宽
            duration (float): 线段显示时间，0 表示永久
        """
        current_pos, _ = p.getBasePositionAndOrientation(self.id)

        # 若位置有更新，则记录轨迹并绘制线段
        if len(self.trajectory) == 0 or not np.allclose(current_pos, self.trajectory[-1]):
            if len(self.trajectory) > 0:
                start = self.trajectory[-1]
                end = current_pos
                color = self.color[:3] if hasattr(self, 'color') else [0, 1, 0]
                p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=width, lifeTime=duration)
            self.trajectory.append(current_pos)
 
    def get_state(self) -> DroneState:
        """
        获取无人机的完整状态，包括位置、朝向、速度等信息

        返回:
        - DroneState: 包含 position、orientation、euler、linear_velocity、angular_velocity、min_distance_to_obstacle 和 collided
        """
        # Get position and orientation from the physics engine
        pos, ori = p.getBasePositionAndOrientation(self.id)
        
        # Get linear and angular velocities from the physics engine
        linear, angular = p.getBaseVelocity(self.id)
        
        # Compute the minimum distance to the nearest obstacle
        min_distance_to_obstacle, _ = self.compute_nearest_obstacle_distance()
        
        # Check if the drone has collided (distance to obstacle is below a threshold)
        collided = min_distance_to_obstacle < 2.0
        
        # Return the drone's state as a DroneState object
        return DroneState(
            position=np.array(pos),                # Position as a numpy array
            orientation=np.array(ori),            # Orientation as a numpy array (quaternion)
            euler=np.array(p.getEulerFromQuaternion(ori)),  # Euler angles from orientation
            linear_velocity=np.array(linear),     # Linear velocity as a numpy array
            angular_velocity=np.array(angular),   # Angular velocity as a numpy array
            min_distance_to_obstacle=np.array(min_distance_to_obstacle),  # Minimum distance to obstacle
            collided=collided                      # Whether the drone has collided with an obstacle
        )

    def update_state(self):
        self.state = self.get_state()
      
    def get_depth_image(self, fov=90, width=224, height=224, near=0.5, far=100.0):
        """
        获取深度图
        
        参数:
            view_matrix: 相机视角矩阵
            projection_matrix: 相机投影矩阵
            width: 图像宽度
            height: 图像高度
        
        返回:
            depth_image: 深度图（归一化为0-1范围）
        """
        # 获取无人机位置与朝向
        pos, orn = p.getBasePositionAndOrientation(self.id)  # 世界坐标下的位姿
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # 定义相机的位置与朝向（以机体坐标为参考）
        local_camera_offset = np.array([0.3, 0.0, -0.2])   # 相机在无人机坐标系中的偏移（机头前方上方）
        local_camera_forward = np.array([1.0, 0.0, 0.0])  # 相机朝向（机头方向）
        local_camera_up = np.array([0.0, 0.0, 1.0])       # 相机“上方”方向（垂直向上）

        # 转换为世界坐标
        camera_eye = np.array(pos) + rot_mat.dot(local_camera_offset)
        camera_target = camera_eye + rot_mat.dot(local_camera_forward) # 无人机注视的目标位置
        camera_up = rot_mat.dot(local_camera_up)

        # 设置视图矩阵和投影矩阵
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, camera_up.tolist())
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, aspect=width / height, nearVal=near, farVal=far)

        # 获取图像信息
        img_arr = p.getCameraImage(width, height, viewMatrix=self.view_matrix, projectionMatrix=self.projection_matrix)

        # 获取深度图信息
        depth_image = np.array(img_arr[3])  # 深度图
        # 获取深度图真实信息
        depth_real = far * near / (far - (far - near) * depth_image)
        depth_normalized = (depth_real - near) / (far - near)
        depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
        
        # self.save_depth_map(depth_normalized)
        return depth_normalized
    
    def get_depth_image_at_angle(self,
                                angle_rad: float,
                                fov: float = 90,
                                width: int = 224,
                                height: int = 224,
                                near: float = 0.5,
                                far: float = 100.0) -> np.ndarray:
        """
        给定一个水平角度 angle_rad（弧度），让摄像头沿该水平方向对准，然后拍摄并返回归一化后的深度图。

        Args:
            angle_rad (float): 在机体坐标系的水平面内，摄像头前向相对于机头正前方偏转的角度（弧度）。
            fov (float):          相机视野 FOV，单位为度（默认为 90°）。
            width (int):          返回图像的宽度（像素）。
            height (int):         返回图像的高度（像素）。
            near (float):         相机近平面距离。
            far (float):          相机远平面距离。

        Returns:
            np.ndarray: 大小 (height, width) 的深度图数组，值已归一化到 [0,1]。
        """
        # 1. 获取无人机在世界坐标系下的位置 pos 和朝向四元数 orn
        pos, orn = p.getBasePositionAndOrientation(self.id)
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

        # 2. 机体坐标系下，摄像头眼睛位置的固定偏移
        local_camera_offset = np.array([0.0, 0.0, -0.5])   # 摄像头比无人机几何中心低 0.5 m
        # 3. 在机体坐标系下，定义一个“纯水平”的前向向量：
        #    这里 [1,0,0] 是机头朝向，在水平面内绕 z 轴旋转 angle_rad
        local_forward = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])

        # 4. 将“机体坐标系”的偏移/前向投影到世界坐标系
        camera_eye = np.array(pos) + rot_mat.dot(local_camera_offset)
        camera_forward_world = rot_mat.dot(local_forward)
        camera_target = camera_eye + camera_forward_world

        # 5. 机体坐标系下的“上方”向量始终取 [0,0,1]
        local_camera_up = np.array([0.0, 0.0, 1.0])
        camera_up = rot_mat.dot(local_camera_up)

        # 6. 计算视图矩阵和投影矩阵
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_eye.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=camera_up.tolist()
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=width / height,
            nearVal=near,
            farVal=far
        )

        # 7. 调用 PyBullet 接口获取 RGBA、深度、分段掩码等信息，img_arr[3] 就是深度缓冲
        img_arr = p.getCameraImage(
            width, height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL  # 或者其他渲染模式
        )

        # img_arr 的格式： (width, height, rgba, depthBuffer, segmentationMask)
        depth_buffer = np.array(img_arr[3], dtype=np.float32)

        # 8. 将深度缓冲（0-1）转换到真实深度值：
        #    depth_real = far * near / (far - (far - near) * depth_buffer)
        # 公式来源：PyBullet文档
        depth_real = far * near / (far - (far - near) * depth_buffer)

        # 9. 归一化到 [0,1] 范围；近处对应 0，远处对应 1
        depth_normalized = (depth_real - near) / (far - near)
        depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

        return depth_normalized

    def compute_nearest_obstacle_distance(self):
        max_check_distance = 10.0  # 最远检测范围（例如 20 米）
        min_distance = 10.0
        nearest_info = None

        for body_id in [i for i in range(p.getNumBodies()) if i != self.id]:
            closest_points = p.getClosestPoints(
                bodyA=self.id,
                bodyB=body_id,
                distance=max_check_distance
            )

            for point in closest_points:
                distance = point[8]
                if distance < min_distance:
                    min_distance = distance
                    nearest_info = {
                        "id": body_id,
                        "name": p.getBodyInfo(body_id)[1].decode('utf-8'),
                        "distance": distance,
                        "position": point[6]
                    }

        if nearest_info:
            return nearest_info["distance"], nearest_info
        else:
            return 10, None
    
    def set_position(self, position):
        p.resetBasePositionAndOrientation(self.id, position, [0, 0, 0, 1])

    def is_heading_aligned_with_velocity(self, velocity, heading, tolerance=1e-3, angle_threshold_deg=1.0):
        """
        检查无人机的头部方向是否与速度方向一致。
        
        参数：
            velocity (np.ndarray): 无人机的速度向量 (vx, vy, vz)，例如 [1.0, 0.0, 0.0]。
            heading (np.ndarray): 无人机的头部方向向量 (即机体 x 轴方向)，例如 [1.0, 0.0, 0.0]。
            tolerance (float): 速度的最小阈值，小于此值认为速度为零。
            angle_threshold_deg (float): 夹角阈值（单位：度），如果夹角小于该值，则认为方向一致。
            
        返回：
            bool: 如果头部方向和速度方向一致，则返回 True，否则返回 False。
        """
        
        # 计算速度的水平分量
        horizontal_velocity = np.array([velocity[0], velocity[1], 0.0])
        horizontal_distance = np.linalg.norm(horizontal_velocity)
        
        if horizontal_distance < tolerance:
            # 如果水平速度太小（接近零），认为无人机处于静止状态
            return True
        
        # 将速度向量归一化
        velocity_direction = horizontal_velocity / horizontal_distance
        
        # 归一化头部方向
        heading_direction = heading / np.linalg.norm(heading)
        
        # 计算速度方向和头部方向之间的夹角
        cos_angle = np.dot(velocity_direction, heading_direction)
        
        # 避免浮动误差，确保角度不会超过1
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # 计算夹角（弧度转度）
        angle = np.arccos(cos_angle) * (180.0 / np.pi)
        
        # 检查夹角是否小于阈值
        return angle < angle_threshold_deg

    def save_depth_map(depth_normalized: np.ndarray, save_dir: str = "output/depth_maps"):
        """
        可视化并保存归一化的深度图。

        参数:
            depth_normalized (np.ndarray): 归一化到 [0, 1] 范围的深度图。
            save_dir (str): 保存目录，默认为 "output/depth_maps"。
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 获取当前时间字符串，格式：20250516_153025（年月日_时分秒）
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 拼接文件名，带时间戳
        save_path = os.path.join(save_dir, f"depth_map_gray_{time_str}.png")

        # 可视化并保存
        plt.figure(figsize=(8, 6))
        plt.imshow(depth_normalized, cmap='gray')  # 使用灰度色图
        plt.colorbar(label='Depth (normalized)')
        plt.title('Normalized Depth Map')
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"深度图已保存至: {save_path}")
    
    def remove_model(self):
        p.removeBody(self.id)

    def remove(self):
        p.removeBody(self.id)

    def check_collision(self, threshold=2.0):
        """
        检查是否碰撞：若与任一障碍物的最近距离小于阈值，则认为发生碰撞并将无人机标记为死亡。

        参数：
            threshold (float): 判定碰撞的距离阈值，单位米，默认2.0

        返回：
            bool: 是否发生碰撞
        """
        distance, nearest_info = self.compute_nearest_obstacle_distance()
        
        if distance < threshold:
            logging.warning(
                f"💥 碰撞检测！障碍物 [ID:{nearest_info['id']}] {nearest_info['name']} 距离 {distance:.2f}m，判定碰撞"
            )
            return True, nearest_info

        return False, None