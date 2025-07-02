import numpy as np
import torch
import os
from torch.utils.tensorboard import SummaryWriter

class RUNNER:
    def __init__(self, agent, env, par, device, mode = 'evaluate', tb_writer: SummaryWriter = None):
        self.agent = agent
        self.env = env
        self.par = par

        # 将 agent 的模型放到指定设备上
        for agent in self.agent.agents.values():
            agent.actor.to(device)
            agent.target_actor.to(device)
            agent.critic.to(device)
            agent.target_critic.to(device)
        
        # 新增：记录传入的 TensorBoard writer（可以为 None）
        self.tb_writer = tb_writer


    def train(self):
        print("=== Training started ===\n")
        step = 0  # 实际与环境交互的步数
        update_count = 0
        captured_count = 0
        time_limit_count = 0
        collided_count = 0

        for episode in range(self.par.episode_num):
            obs = self.env.reset()
            done = {agent_id: False for agent_id in self.env.agents}
            agent_reward = {agent_id: 0.0 for agent_id in self.env.agents}
            while not any(done.values()):
                step += 1
                # ——— 采样动作 ———
                if step < self.par.random_steps:
                    action = {aid: self.env.action_spaces[aid].sample() for aid in self.env.agents}
                else:
                    action = self.agent.select_action(obs)

                # ——— 环境交互 ———
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = {aid: bool(terminated[aid] or truncated[aid]) for aid in self.env.agents}

                # ——— 存储到 Replay Buffer ———
                self.agent.add(obs, action, reward, next_obs, done)

                # ——— 累加 episode 奖励 ———
                for aid, r in reward.items():
                    agent_reward[aid] += r

                # ——— 学习阶段 ———
                if (step >= self.par.random_steps) and (step % self.par.learn_interval == 0):
                    # 调用 learn()，它返回一个 dict: {agent_id: {'critic_loss':..., 'actor_loss':..., 'actor_loss_pse':...}, ...}
                    losses_dict = self.agent.learn(self.par.batch_size, self.par.gamma)
                    # 同步 soft update 目标网络
                    self.agent.update_target(self.par.tau)
                    # 累加“更新次数”
                    update_count += 1

                    # —— 在 TensorBoard 上记录每个 agent 的 loss —— 
                    # step 作为 x 轴
                    for aid, loss_vals in losses_dict.items():
                        # loss_vals: {'critic_loss': float, 'actor_loss': float, 'actor_loss_pse': float}
                        self.tb_writer.add_scalar(f"{aid}/critic_loss",   loss_vals['critic_loss'],    step)
                        self.tb_writer.add_scalar(f"{aid}/actor_loss",    loss_vals['actor_loss'],     step)
                        self.tb_writer.add_scalar(f"{aid}/actor_loss_pse",loss_vals['actor_loss_pse'], step)
                    
                    # ——— 每隔 N 次迭代 保存一次 checkpoint —— 
                    if update_count % self.par.checkpoint_interval == 0:
                        # 直接调用 MADDPG.save_model()，它已经会依次把所有 sub-agent 的 actor/critic/target 保存到文件里
                        print(f"[Runner] Update {update_count} reached. Saving checkpoint...")
                        # 去除多余空格并正确拼接 update_count
                        save_path = os.path.join(self.par.chkpt_dir, f"model_update_{update_count}")
                        self.agent.save_model(save_path)
                    
                    # ——— Evaluate model every eval_interval updates ———
                    if update_count % self.par.eval_interval == 0:
                        # Assume evaluate() returns a scalar (e.g., average validation reward)
                        eval_metric = self.evaluate()
                        print(f"[Evaluation after {update_count} updates] Metric: {eval_metric:.4f}")
                        if self.tb_writer is not None:
                            # Log evaluation metric to TensorBoard under a separate tag
                            self.tb_writer.add_scalar("evaluation/metric", eval_metric, update_count)


                obs = next_obs
            # —— 统计三个特殊事件 —— 
            # 1. “captured” 事件：假设 info["captured"] 是布尔值或非空列表，表示本回合有捕获发生
            if any(info.get("captured").values()):
                captured_count += 1

            # 2. “reached_time_limit” 事件：假设 info["reached_time_limit"] 为 True 时表示本回合因为超步数被截断
            if any(info.get("reached_time_limit").values()):
                time_limit_count += 1

            # 3. “collided” 事件：假设 info["collided"] 是布尔值或非空列表，表示本回合有碰撞发生
            if any(info.get("collided").values()):
                collided_count += 1
            
            # 评估所有回合结束后，计算各事件比率
            N = episode+1
            captured_rate   = captured_count   / N
            time_limit_rate = time_limit_count / N
            collided_rate   = collided_count   / N

            # 在控制台打印
            print(f"[Episode {episode + 1}] captured_rate   = {captured_rate   * 100:.2f}% "
                f"({int(captured_count)}/{int(N)})")
            print(f"[Episode {episode + 1}] time_limit_rate = {time_limit_rate * 100:.2f}% "
                f"({int(time_limit_count)}/{int(N)})")
            print(f"[Episode {episode + 1}] collided_rate   = {collided_rate   * 100:.2f}% "
                f"({int(collided_count)}/{int(N)})\n")

            # ——— 一个 Episode 结束后，把每个 agent 的累计奖励写入 TensorBoard —— 
            total_reward = 0.0
            for aid, r in agent_reward.items():
                total_reward += r
                if self.tb_writer is not None:
                    tag = f"reward/agent_{aid}"
                    self.tb_writer.add_scalar(tag, r, episode + 1)
                
                # 直接在控制台打印该 agent 本轮的奖励
                print(f"[Episode {episode + 1}] Agent {aid} reward: {r:.4f}")

            if self.tb_writer is not None:
                self.tb_writer.add_scalar("reward/total_reward", total_reward, episode + 1)
            
            # 在控制台打印总奖励
            print(f"[Episode {episode + 1}] Total reward: {total_reward:.4f}\n")

        # 直接调用 MADDPG.save_model()，它已经会依次把所有 sub-agent 的 actor/critic/target 保存到文件里
        print(f"[Runner] Saving final model...")
        self.agent.save_model()
        # 最后结束，关闭 writer
        self.tb_writer.close()

    def evaluate(self):
        """
        基于 self.par.eval_episode_num 条评估回合，计算：
        - reward_sum_record（每回合总回报列表）
        - episode_rewards（每个智能体每回合的回报数组）
        - captured_rate（回合捕获事件比率）
        - time_limit_rate（回合超时截断比率）
        - collided_rate（回合碰撞事件比率）

        最终将 captured_rate、time_limit_rate、collided_rate 写入 TensorBoard，并返回 success_rate 或其他指标。
        """
        # 重置评估用的数据结构
        self.reward_sum_record = []
        self.episode_rewards = {
            agent_id: np.zeros(self.par.eval_episode_num)
            for agent_id in self.env.agents
        }

        # 三个计数器，用于统计这 par.eval_episode_num 次回合里各事件发生的次数
        captured_count = 0
        time_limit_count = 0
        collided_count = 0

        for episode in range(self.par.eval_episode_num):
            step = 0
            print(f"[Evaluation] Episode {episode + 1}/{self.par.eval_episode_num}")

            reset_result = self.env.reset()
            if isinstance(reset_result, tuple):
                obs, _env_info = reset_result
            else:
                obs = reset_result

            done = {agent_id: False for agent_id in self.env.agents}
            agent_reward = {agent_id: 0.0 for agent_id in self.env.agents}

            # 这一回合内部循环
            while not any(done.values()):
                step += 1
                action = self.agent.select_action(obs)  # 假设为确定性策略
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = {
                    aid: bool(terminated[aid] or truncated[aid])
                    for aid in self.env.agents
                }

                for aid, r in reward.items():
                    agent_reward[aid] += r
                obs = next_obs

            # 回合结束后，统计总回报、每智能体回报
            sum_reward = sum(agent_reward.values())
            self.reward_sum_record.append(sum_reward)
            for aid, r in agent_reward.items():
                self.episode_rewards[aid][episode] = r

            # —— 统计三个特殊事件 —— 
            # 1. “captured” 事件：假设 info["captured"] 是布尔值或非空列表，表示本回合有捕获发生
            if any(info.get("captured").values()):
                captured_count += 1

            # 2. “reached_time_limit” 事件：假设 info["reached_time_limit"] 为 True 时表示本回合因为超步数被截断
            if any(info.get("reached_time_limit").values()):
                time_limit_count += 1

            # 3. “collided” 事件：假设 info["collided"] 是布尔值或非空列表，表示本回合有碰撞发生
            if any(info.get("collided").values()):
                collided_count += 1

        # 评估所有回合结束后，计算各事件比率
        N = float(self.par.eval_episode_num)
        captured_rate   = captured_count   / N
        time_limit_rate = time_limit_count / N
        collided_rate   = collided_count   / N

        # 在控制台打印
        print(f"\n[Evaluation] captured_rate   = {captured_rate   * 100:.2f}% "
            f"({int(captured_count)}/{int(N)})")
        print(f"[Evaluation] time_limit_rate = {time_limit_rate * 100:.2f}% "
            f"({int(time_limit_count)}/{int(N)})")
        print(f"[Evaluation] collided_rate   = {collided_rate   * 100:.2f}% "
            f"({int(collided_count)}/{int(N)})\n")

        # —— 记录到 TensorBoard —— 
        # 假设你已经在初始化时传入了 self.tb_writer = SummaryWriter(log_dir=...)
        if self.tb_writer is not None:
            # global_step 或者你觉得合适的横坐标（可以用 update_count、episode、self.global_step 等）
            step_for_tb = self.global_step if hasattr(self, "global_step") else episode + 1

            # 将比率写为标量
            self.tb_writer.add_scalar("evaluation/captured_rate",   captured_rate,   step_for_tb)
            self.tb_writer.add_scalar("evaluation/time_limit_rate", time_limit_rate, step_for_tb)
            self.tb_writer.add_scalar("evaluation/collided_rate",   collided_rate,   step_for_tb)

            # 如果你还想把“平均总回报”也一起存
            avg_total_reward = np.mean(self.reward_sum_record)
            self.tb_writer.add_scalar("evaluation/avg_total_reward", avg_total_reward, step_for_tb)

        # 返回某个你需要的指标（这里以成功率举例，实际可自行替换）
        success_rate = 1.0 - time_limit_rate  # 例如：只要不是超时就算“成功”，你可改为别的逻辑
        return success_rate
