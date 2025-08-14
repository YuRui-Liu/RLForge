import time
import sys
import json
from collections import deque
import re
sys.path.append(r"E:\\CMVC\\Proj\\usvlib4ros_origin")
import torch
import numpy as np
from algor.ppo import PPOAgent
import os
from usvlib4ros.usvRosUtil import LogUtil, USVRosbridgeClient
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# 连续动作转离散动作（线性映射）
def continuous_to_discrete(action, num_actions=5):
    """将[-1,1]的连续动作映射到[0,4]的离散动作"""
    # 缩放到[0,1]
    scaled = (action + 1) / 2.0
    # 映射到离散索引
    discrete_idx = int(scaled * (num_actions - 1))
    # 确保在有效范围内
    return max(0, min(discrete_idx, num_actions - 1))

# ====================================================================================================
# PPO 训练函数 
# ====================================================================================================

def train_ppo_agent(
        total_timesteps,
        state_dim,
        action_dim,
        hidden_dim,
        lr,
        gamma,
        gae_lambda,
        clip_param,
        ppo_epochs,
        ent_coef,
        rollout_steps,  # PPO特有参数：每次更新前收集的步数
        save_dir,
        log_dir,
        resume_model_path=None,
        initial_model_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_interval=100,  # 每隔多少个回合记录一次平均日志
        save_interval=100,  # 每隔多少个回合保存一次模型
        episode_step_limit=800,  # 每个回合的最大步数
):
    # 处理模型路径和恢复训练逻辑
    if resume_model_path:
        # 如果是恢复训练，尝试从路径中解析方法名并设置日志/保存目录
        if os.path.exists(resume_model_path):
            match = re.search(r"results[\\/]([A-Za-z0-9@_/-]+)", resume_model_path)
            if match:
                method_name = match.group(1)
                print(f"加载检查点: {method_name}")
                base_results_path = os.path.dirname(os.path.dirname(resume_model_path))
                log_dir = os.path.join(base_results_path, method_name)
                save_dir = os.path.join(log_dir, "checkpoints")
                print(f"模型保存位置: {save_dir}")
            else:
                print(f"警告: 无法从 {resume_model_path} 解析方法名，使用默认日志/保存目录。")
        else:
            print(f"警告: 恢复模型路径 {resume_model_path} 不存在。将从头开始训练。")
            resume_model_path = None  # 如果路径无效，则不加载模型

    # --- 环境和智能体初始化 --- #
    # 【....自己的环境....】

    # --- 初始化 PPO 智能体 --- #
    ppo_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_param=clip_param,
        ppo_epochs=ppo_epochs,
        ent_coef=ent_coef
    )

    # 尝试加载模型如果 resume_model_path 有效
    start_epoch = 0
    global_step = 0
    if resume_model_path and os.path.exists(resume_model_path):
        print(f"正在从 {resume_model_path} 加载模型...")
        try:
            checkpoint = torch.load(resume_model_path, map_location=device)
            ppo_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            ppo_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            ppo_agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1  # 从下一回合开始
            global_step = checkpoint.get('global_step', 0)  # 恢复全局步数
            print(f"模型加载成功。从回合 {start_epoch}, 全局步数 {global_step} 继续训练。")
        except Exception as e:
            print(f"加载模型失败: {e}。将从头开始训练。")
            resume_model_path = None  # 加载失败则不使用旧模型



    # 创建日志目录和模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # 初始化变量
    episode_rewards = deque(maxlen=log_interval)  # 只保留最近 log_interval 个回合的奖励
    episode_lengths = deque(maxlen=log_interval)  # 只保留最近 log_interval 个回合的长度
    episode_count = start_epoch
    start_time = time.time()

    print(f"开始训练，总步数: {total_timesteps}, 使用设备: {device}")

    current_state = env.reset()  # 获取初始状态
    done_episode = False  # 标记当前回合是否已结束

    while global_step < total_timesteps:
        # 1. **数据收集阶段 (Rollout Collection)**
        # 收集一个rollout的经验数据
        states_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        rewards_buffer = []
        dones_buffer = []

        # 如果上一个回合已经结束，则重置环境开启新回合
        if done_episode:
            current_state = env.reset()
            done_episode = False  # 重置回合结束标志

        # 记录当前rollout/episode的奖励和长度，用于临时累加
        rollout_episode_reward = 0
        rollout_episode_length = 0

        # 在当前episode中或直到收集到足够rollout_steps步为止
        for step_in_rollout in range(rollout_steps):
  

            # 智能体选择动作
            action, log_prob = ppo_agent.select_action(current_state)

            # 存储当前时间步的数据
            states_buffer.append(current_state)
            actions_buffer.append(action.cpu().numpy())  # 动作需要转回Numpy
            log_probs_buffer.append(log_prob.cpu().numpy())

            discrete_action = continuous_to_discrete(action.cpu().numpy()[0])
            # 环境执行动作
            next_state, reward, done = env.step(current_state, discrete_action)

            # 存储环境反馈
            rewards_buffer.append(reward)
            dones_buffer.append(float(done))  # done转换为浮点数0.0或1.0

            # 更新当前状态和回合统计
            current_state = next_state
            rollout_episode_reward += reward
            rollout_episode_length += 1
            global_step += 1

            # 检查回合是否结束或达到最大步数
            if done or rollout_episode_length >= episode_step_limit:
                done_episode = True  # 标记当前回合已结束
                # 如果回合结束，则当前rollout提前结束
                break

            # 2. **计算最后一个状态的价值 (用于GAE)**
            final_state_value = 0
            # 如果rollout不是因为回合结束而中断（即达到了rollout_steps限制），
            # 则需要计算当前状态的价值用于GAE的引导。
            if not done_episode:
                with torch.no_grad():
                    final_state_value = ppo_agent.critic(
                        torch.FloatTensor(current_state).unsqueeze(0).to(device)).item()

            # 确保收集到了数据才进行更新
            if len(states_buffer) > 0:
                # 3. **PPO 策略更新阶段**
                actor_loss, critic_loss, total_loss, entropy_loss = ppo_agent.update(
                    (states_buffer, actions_buffer, log_probs_buffer, rewards_buffer, dones_buffer, final_state_value)
                )

                # 记录损失到TensorBoard
                writer.add_scalar("Train/Actor_Loss", actor_loss, global_step)
                writer.add_scalar("Train/Critic_Loss", critic_loss, global_step)
                writer.add_scalar("Train/Total_Loss", total_loss, global_step)
                writer.add_scalar("Train/Entropy_Loss", entropy_loss, global_step)
            else:
                actor_loss, critic_loss, total_loss, entropy_loss = 0, 0, 0, 0  # 没有数据则损失为0

            # 4. **回合统计和日志**
            # 如果在当前的rollout中有一个回合完成了
            if done_episode:
                episode_rewards.append(rollout_episode_reward)
                episode_lengths.append(rollout_episode_length)

                writer.add_scalar("Episode/reward", rollout_episode_reward, episode_count)
                writer.add_scalar("Episode/length", rollout_episode_length, episode_count)


                episode_count += 1  # 回合计数增加

                # 每隔 log_interval 个回合记录平均统计信息
                if episode_count % log_interval == 0:
                    avg_reward = np.mean(episode_rewards)
                    avg_length = np.mean(episode_lengths)
                    print(
                        f"Episode: {episode_count}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Step: {global_step}"
                    )
                    writer.add_scalar("Reward/Cumulative_Epoch_Reward", avg_reward, episode_count)
                    writer.add_scalar("Reward/Cumulative_Step_Reward", avg_reward, global_step)
                    writer.add_scalar("Reward/Avg_Length_Last_X_Episodes", avg_length, episode_count)  # 记录平均长度

                # 每隔 save_interval 个回合保存模型
                if episode_count % save_interval == 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{episode_count}.pt")
                    torch.save({
                        'epoch': episode_count,
                        'global_step': global_step,
                        'actor_state_dict': ppo_agent.actor.state_dict(),
                        'critic_state_dict': ppo_agent.critic.state_dict(),
                        'optimizer_state_dict': ppo_agent.optimizer.state_dict(),
                    }, checkpoint_path)
                    print(f"模型保存到 {checkpoint_path}")

        writer.close()
        print(f"训练结束，耗时: {(time.time() - start_time) / 60:.2f} 分钟")


# ====================================================================================================
# 主函数
# ====================================================================================================

def main():
    # 结果保存路径
    results_path = "./results/"  # 建议使用相对路径，更具移植性
    method_name = f"ppo_" #{datetime.now().strftime('%m%d-%H%M')}"
    log_dir = os.path.join(results_path, method_name)  # 使用os.path.join来构建路径

    TRAIN_PARAMS = {
        'initial_model_path': None,  # 初始课程学习模型路径，如果从头开始，设为None
        'resume_model_path': None,  # 继续训练的模型检查点路径，如果从头开始，设为None
        'save_dir': os.path.join(log_dir, "checkpoints"),  # 模型保存目录
        'log_dir': log_dir,  # TensorBoard日志目录
        'state_dim': 97,  # 状态空间维度，请根据您的环境调整
        'action_dim': 1,  # 动作空间维度 (离散动作)
        'hidden_dim': 128,  # 策略和价值网络的隐藏层维度
        "total_timesteps": 8_000_000,  # 总训练步数 (环境交互步数)
        'lr': 3e-4,  # 学习率
        'gamma': 0.99,  # 折扣因子
        'gae_lambda': 0.95,  # GAE (Generalized Advantage Estimation) 参数
        'clip_param': 0.2,  # PPO裁剪参数
        'ppo_epochs': 10,  # 每次收集rollout数据后，进行多少次策略更新
        'ent_coef': 0.01,  # 熵损失系数，鼓励探索
        'rollout_steps': 2048,  # 每次更新前收集的经验步数 (重要PPO参数)
        'episode_step_limit': 1e3,  # 每个回合的最大步数
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 自动选择设备
        'log_interval': 10,  # 每10个回合打印一次平均训练信息
        'save_interval': 100,  # 每100个回合保存一次模型
    }

    # 如果不是从检查点继续训练，则创建新的日志目录并保存配置
    if not TRAIN_PARAMS['resume_model_path']:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(TRAIN_PARAMS, f, ensure_ascii=False, indent=4)  # 将训练参数保存到config.json

    # 调用训练函数
    train_ppo_agent(**TRAIN_PARAMS)


if __name__ == "__main__":
    main()