import numpy as np

class RewardComponent:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def compute(self, env, **kwargs):
        # 默认行为（可被子类覆盖）
        return 0.0

class DistanceReward(RewardComponent):
    def compute(self, chaser_states, runner_state):
        chaser_pos = [state.position for state in chaser_states]
        runner_pos = runner_state.position
        distances = np.linalg.norm(chaser_pos - runner_pos, axis=1)
        return -np.sum(distances) * self.weight
    
class AngleReward(RewardComponent):
    def compute(self, chaser_states, runner_state):
        target_position = np.array(runner_state.position)
        chaser_pos = np.array([s.position for s in chaser_states])

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

