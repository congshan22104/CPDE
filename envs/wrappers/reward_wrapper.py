import numpy as np

class RewardComponent:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def compute(self, env, **kwargs):
        # 默认行为（可被子类覆盖）
        return 0.0

class TargetProgressReward(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        drone_position = env.sim.drone.state.position
        target_position = env.sim.drone.target_position
        distance_to_target = np.linalg.norm(drone_position - target_position)

        if not hasattr(env, "prev_distance_to_target") or env.prev_distance_to_target is None:
            env.prev_distance_to_target = distance_to_target

        improvement = env.prev_distance_to_target - distance_to_target
        env.prev_distance_to_target = distance_to_target

        return improvement  # 缩放一下数值

class NonlinearCollisionPenalty(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        min_distance_to_obstacle = env.sim.drone.state.min_distance_to_obstacle

        if min_distance_to_obstacle >= 10.0:
            return 0.0
        penalty = -np.exp(-(min_distance_to_obstacle - 2))
        return max(penalty, -1.0)* self.weight

class LinearCollisionPenalty(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        min_distance_to_obstacle = env.sim.drone.state.min_distance_to_obstacle

        if min_distance_to_obstacle < 10.0:
            collision_penalty = -(10 - min_distance_to_obstacle) / 10
        else:
            collision_penalty = 0.0
        
        return collision_penalty * self.weight

class ImageLinearCollisionPenalty(RewardComponent):
     def compute(self, env, **kwargs) -> float:
        '''
        机制：
            如果最近障碍物距离大于 15 米，奖励 +1;
            距离在 10-15 米之间时，奖励线性减小;
            距离在 5-10 米之间，惩罚线性增强;
            距离小于 5 米时，惩罚为 -1。
        '''
        obs = kwargs.get("obs", {})
        depth_image = obs

        # 归一化深度图还原实际距离
        min_range = 0.5
        max_range = 100.0
        min_depth_norm = np.min(depth_image)
        min_depth_real = min_depth_norm * (max_range - min_range) + min_range

        if min_depth_real > 15.0:
            reward = 1.0
        elif min_depth_real > 10.0:
            # (15, 25] 奖励从 1 降到 0
            reward = (15.0 - min_depth_real) / 10.0  # x ∈ [0, 1]
        elif min_depth_real > 5.0:
            # (5, 15] 奖励从 0 降到 -1（惩罚）
            reward = -(10.0 - min_depth_real) / 10.0  # x ∈ [0, 1]
        else:
            reward = -1.0

        return reward

class ImageNonlinearCollisionPenalty(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        '''
        机制：
            10米以上给正奖励;
            10米以下给负奖励
        '''
        depth_image = kwargs.get("obs", {})

        if depth_image is None:
            raise ValueError("[ImageNonlinearCollisionPenalty] obs['depth_image'] is None.")

        # 归一化深度图还原实际距离
        min_range = 0.5
        max_range = 100.0
        min_depth_norm = np.min(depth_image)
        min_depth_real = min_depth_norm * (max_range - min_range) + min_range
        
        # 距离小于5时，惩罚最大-1，距离大于15时，奖励接近1
        # tanh在d=5时应该接近-1，在d=15时接近+1，中心在10
        # 参数调整
        center = 10       # tanh中心点在距离10m
        steepness = 0.7   # 控制tanh曲线陡峭程度（越大越陡）
        
        # tanh输出[-1, 1]
        reward = np.tanh(steepness * (min_depth_real - center))
        
        # 输出裁剪为[-1, 1]
        reward = np.clip(reward, -1, 1)
        
        return float(reward)

class ImageNonlinearCollisionPenalty2(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        depth_image = kwargs.get("obs", {})

        if depth_image is None:
            raise ValueError("[ImageNonlinearCollisionPenalty] obs['depth_image'] is None.")

        # 归一化深度图还原实际距离
        min_range = 0.5
        max_range = 100.0
        min_depth_norm = np.min(depth_image)
        min_depth_real = min_depth_norm * (max_range - min_range) + min_range
        
        # 距离小于5时，惩罚最大-1，距离大于15时，奖励接近1
        # tanh在d=5时应该接近-1，在d=15时接近+1，中心在10
        # 参数调整
        center = 20       # tanh中心点在距离20m
        steepness = 0.3   # 控制tanh曲线陡峭程度（越大越陡）
        
        # tanh输出[-1, 1]
        reward = np.tanh(steepness * (min_depth_real - center))
        
        # 输出裁剪为[-1, 1]
        reward = np.clip(reward, -1, 1)
        
        return float(reward)

class DistanceToObstacleReward(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        min_distance = env.sim.drone.state.min_distance_to_obstacle

        if min_distance >= 10.0:
            reward = 1.0
        elif min_distance >= 2.0:
            # 在 [2, 10] 米之间，线性从 -1 过渡到 1
            reward = -1.0 + 2.0 * (min_distance - 2.0) / (10.0 - 2.0)
        else:
            reward = -1.0  # 低于2米视为最危险，给最小奖励

        return reward
    
class DirectionReward(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        drone_position = env.sim.drone.state.position
        drone_velocity = env.sim.drone.state.linear_velocity
        target_position = env.sim.drone.target_position

        if np.linalg.norm(drone_velocity) > 1e-6:
            velocity_dir = drone_velocity / np.linalg.norm(drone_velocity)
            direction_to_target = (target_position - drone_position) / np.linalg.norm(target_position - drone_position)
            cos_theta = np.clip(np.dot(velocity_dir, direction_to_target), -1.0, 1.0)
            direction_reward = cos_theta  # cos越接近1，飞得越对
        else:
            direction_reward = 0.0  # 静止不给奖励

        # 可以调权重，在组件管理里也可以再调
        return direction_reward

class SphericalDirectionReward(RewardComponent):
    def cartesian_to_spherical(self, vec):
        """
        将三维向量转换为球坐标方向角 (theta, phi)
        theta: [0, pi] 与 z 轴夹角
        phi: [-pi, pi] 与 x 轴夹角（在 xy 平面）
        """
        x, y, z = vec
        r = np.linalg.norm(vec)
        if r < 1e-6:
            return 0.0, 0.0  # 避免除零
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return theta, phi

    def compute(self, env, **kwargs) -> float:
        # 获取机器人当前位置和速度
        drone_position = env.sim.drone.state.position
        target_position = env.sim.drone.target_position
        drone_velocity = env.sim.drone.state.linear_velocity  # 获取机器人的速度向量
        # 如果速度非零
        if np.linalg.norm(drone_velocity) > 1e-6:
            # 获取两个方向向量
            velocity_dir = drone_velocity / np.linalg.norm(drone_velocity)
            target_dir = target_position - drone_position
            target_dir /= np.linalg.norm(target_dir)

            # 转换为球坐标方向角
            theta_v, phi_v = self.cartesian_to_spherical(velocity_dir)
            theta_t, phi_t = self.cartesian_to_spherical(target_dir)

            # 角度差归一化为奖励（角度越小越接近）
            delta_theta = abs(theta_v - theta_t) / np.pi        # ∈ [0, 1]
            delta_phi = abs(phi_v - phi_t) / (2 * np.pi)         # ∈ [0, 1]

            # 合成方向奖励（值越小越好，可反转）
            direction_reward = 1.0 - (0.5 * delta_theta + 0.5 * delta_phi)  # ∈ [0, 1]
             # 映射到 [-1, 1]
            direction_reward = 2.0 * direction_reward - 1.0
        else:
            direction_reward = 0.0  # 静止时不给奖励
        
        return direction_reward * self.weight

class CosineSphericalDirectionReward(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        obs = kwargs.get("obs", {})
        state = obs.get("state", None)

        if state is None or len(state) < 2:
            return 0.0

        # 假设 state[-2:] 是方向差 [delta_theta, delta_phi]，已经归一化到 [0, 1]
        # 先反归一化回 [-π, π]
        delta_theta = (state[-2] - 0.5) * 2 * np.pi
        delta_phi   = (state[-1] - 0.5) * 2 * np.pi

        # 只在 -π/2 ~ π/2 范围内给正奖励，超出范围给负奖励
        # 使用 cos 非线性奖励，cos(0) = 1，cos(±π) = -1，平滑过渡
        reward_theta = np.cos(delta_theta)
        reward_phi   = np.cos(delta_phi)

        # 组合奖励（可调权重）
        reward = 0.5 * reward_theta + 0.5 * reward_phi

        return float(reward)

class CosineSphericalDirectionReward2(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        obs = kwargs.get("obs", {})
        state = obs.get("state", None)

        if state is None or len(state) < 2:
            return 0.0

        delta_theta = (state[-2] - 0.5) * 2 * np.pi  # [-π, π]
        delta_theta = np.abs(delta_theta)  # 只考虑水平夹角的绝对值

        lower = 5 / 12 * np.pi  # 75°
        upper = 7 / 12 * np.pi  # 105°

        if delta_theta <= lower:
            reward = 1.0
        elif delta_theta <= upper:
            # 在 [lower, upper] 之间做 cos 插值，使 reward 从 1 降到 0
            ratio = (delta_theta - lower) / (upper - lower)
            reward = np.cos(ratio * np.pi) * 0.5 + 0.5  # cos 从 1 -> -1，缩放到 1 -> 0
        elif delta_theta <= np.pi:
            # 在 [upper, π] 之间做 cos 插值，使 reward 从 0 降到 -1
            ratio = (delta_theta - upper) / (np.pi - upper)
            reward = np.cos(ratio * np.pi) * 0.5 - 0.5  # cos 从 1 -> -1，缩放到 0 -> -1
        else:
            reward = -1.0

        return float(reward)

class TanhSphericalDirectionReward(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        obs = kwargs.get("obs", {})
        state = obs.get("state", None)

        if state is None or len(state) < 2:
            return 0.0

        # 假设 state[-2:] 是方向差 [delta_theta, delta_phi]，已经归一化到 [0, 1]
        # 先反归一化回 [-π, π]
        delta_theta = (state[-2] - 0.5) * 2 * np.pi
        delta_phi   = (state[-1] - 0.5) * 2 * np.pi

        # 只在 -π/2 ~ π/2 范围内给正奖励，超出范围给负奖励
        # 使用 cos 非线性奖励，cos(0) = 1，cos(±π) = -1，平滑过渡
        steepness = 6   # 控制tanh曲线陡峭程度（越大越陡）
        reward_theta = np.tanh(steepness * delta_theta)
        reward_phi   = np.tanh(steepness * delta_phi)

        # 组合奖励（可调权重）
        reward = 0.5 * reward_theta + 0.5 * reward_phi

        return float(reward)
    
class InterpolationSphericalDirectionReward(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        obs = kwargs.get("obs", {})
        state = obs.get("state", None)

        if state is None or len(state) < 2:
            return 0.0

        delta_theta = (state[-2] - 0.5) * 2 * np.pi  # [-π, π]
        delta_theta = np.abs(delta_theta)  # 只考虑水平夹角的绝对值

        lower = 5 / 12 * np.pi  # 75°
        upper = 7 / 12 * np.pi  # 105°

        if delta_theta <= lower:
            reward = 1.0
        elif delta_theta <= upper:
            # 线性插值：从 1 到 0
            ratio = (delta_theta - lower) / (upper - lower)
            reward = 1.0 - ratio
        elif delta_theta <= np.pi:
            # 线性插值：从 0 到 -1
            ratio = (delta_theta - upper) / (np.pi - upper)
            reward = -ratio
        else:
            reward = -1.0

        return float(reward)

class NonlinearSphericalDirectionReward(RewardComponent):
    def cartesian_to_spherical(self, vec):
        """
        将三维向量转换为球坐标方向角 (theta, phi)
        theta: [0, pi] 与 z 轴夹角
        phi: [-pi, pi] 与 x 轴夹角（在 xy 平面）
        """
        x, y, z = vec
        r = np.linalg.norm(vec)
        if r < 1e-6:
            return 0.0, 0.0  # 避免除零
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return theta, phi

    def compute(self, env, **kwargs) -> float:
        drone_position = env.sim.drone.state.position
        target_position = env.sim.drone.target_position
        drone_velocity = env.sim.drone.state.linear_velocity

        if np.linalg.norm(drone_velocity) < 1e-6:
            return 0.0  # 静止时不给奖励

        # 单位方向向量
        velocity_dir = drone_velocity / np.linalg.norm(drone_velocity)
        target_dir = target_position - drone_position
        target_dir /= np.linalg.norm(target_dir)

        # 转换为球坐标角度 (θ, φ)
        theta_v, phi_v = self.cartesian_to_spherical(velocity_dir)
        theta_t, phi_t = self.cartesian_to_spherical(target_dir)

        # 计算两个角度差
        delta_theta = abs(theta_v - theta_t)
        delta_phi = abs(phi_v - phi_t)
        # 归一化到 [-π, π] 区间
        if delta_phi > np.pi:
            delta_phi = 2 * np.pi - delta_phi

        # 计算余弦作为奖励（∈ [-1, 1]）
        reward_theta = np.cos(delta_theta)  # θ方向对齐程度
        reward_phi   = np.cos(delta_phi)   # φ方向对齐程度

        # 组合方向奖励，等权平均
        direction_reward = 0.5 * reward_theta + 0.5 * reward_phi

        return direction_reward * self.weight

class VelocityReward(RewardComponent):
    def compute(self, env, **kwargs) -> float:
        """
        计算速度奖励 rvel，根据目标位置 Pg、机器人当前位置 Pr 和机器人速度 Vr
        
        参数:
        - env: 环境对象，包含无人机和目标的位置与速度

        返回:
        - rvel: 速度奖励值
        """
        # 获取机器人当前位置和速度
        drone_position = env.sim.drone.state.position
        target_position = env.sim.drone.target_position
        velocity = env.sim.drone.state.linear_velocity  # 获取机器人的速度向量

        # 计算目标方向向量
        direction_vector = target_position - drone_position

        # 计算目标方向与速度方向的点积
        dot_product = np.dot(direction_vector, velocity)

        # 计算目标方向的欧几里得距离
        distance_to_goal = np.linalg.norm(direction_vector)

        # 如果机器人已经在目标位置，避免除零错误
        if distance_to_goal == 0:
            return 0

        # 计算速度奖励：速度对齐程度越高，奖励越大，速度越大，奖励越大
        velocity_reward = dot_product / distance_to_goal  # 根据速度对齐目标方向的程度来奖励

        return velocity_reward

class TerminalReward(RewardComponent):
    def __init__(self, name, weight, arrival_reward=100.0, collision_penalty=-100.0):
        self.name = name
        self.weight = weight
        self.arrival_reward = arrival_reward
        self.collision_penalty = collision_penalty
    def compute(self, env, **kwargs):
        is_arrived = kwargs.get("is_arrived", False)
        is_collided = kwargs.get("is_collided", False)
        if is_arrived:
            return self.arrival_reward
        elif is_collided:
            return self.collision_penalty
        return 0.0

