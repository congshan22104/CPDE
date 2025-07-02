import numpy as np
import pybullet as p

class RewardComponent:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def compute(self, env, **kwargs):
        # 默认行为（可被子类覆盖）
        return 0.0

class DistanceReward(RewardComponent):
    def compute(self, chaser, runner):
        chaser_pos = chaser.state.position
        runner_pos = runner.state.position
        distances = np.linalg.norm(chaser_pos - runner_pos)
        return -np.sum(distances) * self.weight
    
class AngleReward(RewardComponent):
    def compute(self, chaser, runner):
        target_position = np.array(runner.state.position)
        chaser_pos = np.array([s.position for s in chaser.states])

        # 计算每架无人机的绝对方位角
        angles = np.arctan2(
            chaser_pos[:, 1] - target_position[1],
            chaser_pos[:, 0] - target_position[0]
        ) * 180 / np.pi
        angles = np.mod(angles, 360)

        # 计算相邻两点的环绕最小角度差
        num = len(chaser_pos)
        angle_diff = np.zeros(num)
        for i in range(num):
            j = (i + 1) % num
            d = abs(angles[i] - angles[j])
            angle_diff[i] = min(d, 360 - d)

        # 理想间隔
        ideal = 360.0 / num

        # 计算总偏差和最大偏差
        penalty = np.sum(np.abs(angle_diff - ideal))
        max_penalty = num * ideal

        # 使用 tanh 函数归一化奖励
        scaled_penalty = penalty / max_penalty  # 缩放偏差
        normalized_reward = np.tanh(5 * (1 - scaled_penalty))  # 乘以常数因子来控制惩罚强度

        # 最终奖励值，根据权重调整
        return normalized_reward * self.weight

class ObstacleClearanceReward(RewardComponent):
    def compute(self, chaser, runner):
        position = chaser.target_position
        ignore_ids = [runner.id, chaser.id]
        distance = self.compute_nearest_distance_to_obstacle(position, ignore_ids)

        # 奖励函数参数
        # distance=6.0, reward=-0.999
        # distance=8.0, reward=-0.964
        # distance=9.0, reward=-0.761
        # distance=10.0, reward= 0.000
        # distance=11.0, reward= 0.761
        # distance=12.0, reward= 0.964
        # distance=14.0, reward= 0.999
        k = 1.0
        c = 10.0

        # 非线性奖励：距离越大越接近 +1，越小越接近 -1
        reward = np.tanh(k * (distance - c))


        return reward
    
    def compute_nearest_distance_to_obstacle(self, position, ignore_ids=None, radius=0.1, max_distance=20.0):
        """
        在指定位置创建一个小球体，然后计算它到最近障碍物的距离。

        Args:
            position (tuple): 球体中心坐标 (x, y, z)
            ignore_ids (list): 要忽略的 bodyUniqueId 列表，例如自身无人机的 ID
            radius (float): 球体半径
            max_distance (float): 查询的最大距离

        Returns:
            float: 到最近障碍物的距离，如果没有检测到障碍物，则返回 max_distance
        """
        if ignore_ids is None:
            ignore_ids = []

        # 创建一个球体用于检测
        col_shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual_shape_id = -1  # 不需要可视化

        ball_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )

        # 初始化最小距离为最大值
        min_distance = max_distance

        # 遍历所有物体
        for i in range(p.getNumBodies()):
            body_id = p.getBodyUniqueId(i)
            if body_id == ball_id or body_id in ignore_ids:
                continue
            closest_points = p.getClosestPoints(ball_id, body_id, distance=max_distance)
            for cp in closest_points:
                if cp[8] < min_distance:
                    min_distance = cp[8]

        # 删除探测球体
        p.removeBody(ball_id)

        return min_distance

