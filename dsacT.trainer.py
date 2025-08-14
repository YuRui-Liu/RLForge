import time
import sys
from collections import deque
import torch
import numpy as np
from algor.dsacv2 import map_continuous_to_discrete_better, DSACv2
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import re
import json

sys.path.append(r"E:\\CMVC\\Proj\\usvlib4ros_origin")

MAX_EPOCH = 10000

OVER_EPOCH = 9001

def train_dsacv2_agent(
        total_timesteps,
        dim_states,
        dim_actions,  # 这将是连续动作空间的维度 (DSACv2网络的输出维度)
        use_cnn,
        use_lstm,
        tau,
        tau_b,
        actor_lr,
        critic_lr,
        alpha_lr,
        gamma,
        batch_size,
        memory_capacity,
        alpha,
        auto_alpha,
        target_entropy,
        gradient_adjustment_eta,
        lstm_hidden_size,
        lstm_layers,
        save_dir,
        log_dir,
        train_freq=4,
        resume_model_path=None,
        initial_model_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_interval=100,
        save_interval=1000,
        episode_step_limit=800,
):
    model_path_type = None
    if resume_model_path is None:
        if initial_model_path is None:
            pass
        else:
            resume_model_path = initial_model_path
            model_path_type = "course"
            print("加载课程学习:", resume_model_path)
    else:
        # 注意：这里需要确保resume_model_path包含类似 "results\\method_name\\" 的结构
        match = re.search(r"results[\\/]([A-Za-z0-9@_/-]+)", resume_model_path)
        if match:
            method_name = match.group(1)
            print("加载检查点:", method_name)
            base_results_path = r"E:\CMVC\Proj\usvlib4ros_origin\results"
            log_dir = os.path.join(base_results_path, method_name)
            save_dir = os.path.join(log_dir, "checkpoints")
            print("模型保存位置:", save_dir)
            model_path_type = "resume"
        else:
            print(f"警告: 无法从 {resume_model_path} 解析 method_name。将使用传入的log_dir/save_dir。")
            # 如果解析失败，则使用传入的 log_dir 和 save_dir，它们应在 main() 中设置

    # --- 环境和智能体初始化 --- #
    # env_discrete_action_size = 
    # env = [... 使用自己的环境...]

    # dsac_agent的model_path和model_path_type传递
    dsac_agent = DSACv2(
        model_path=resume_model_path,  # 传入解析后的模型路径
        # model_path_type=model_path_type,  # 传入模型路径类型
        use_cnn=use_cnn,
        use_lstm=use_lstm,
        dim_states=dim_states,
        dim_actions=dim_actions,  # DSACv2 的 dim_actions 是连续动作空间的维度
        tau=tau,
        tau_b=tau_b,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        alpha_lr=alpha_lr,
        gamma=gamma,
        batch_size=batch_size,
        memory_capacity=memory_capacity,
        alpha=alpha,
        auto_alpha=auto_alpha,
        target_entropy=target_entropy,
        gradient_adjustment_eta=gradient_adjustment_eta,
        cnn_tar_dim=90,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        device=device
    )


    # 这些目录应该在 main 函数中根据 resume_model_path 决定并传入
    # 但如果 main 函数没有处理，这里也应该确保它们存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 确保 log_dir 也被创建

    writer = SummaryWriter(log_dir=log_dir)

    # 初始化变量
    episode_rewards = deque(maxlen=log_interval)
    episode_lengths = deque(maxlen=log_interval)
    global_step = dsac_agent.total_steps
    episode_count = dsac_agent.start_epoch
    start_time = time.time()

    print(f"开始训练 DSAC-T，总步数: {total_timesteps}, 使用设备: {device}")

    while global_step < total_timesteps:
        state = env.reset() 
        done = False
        episode_reward = 0
        episode_length = 0

        step_times = []  # 新增：记录每步耗时
        a_times, s_times = [], []
        l_times = []
        # 在每个 episode 开始时重置 LSTM 状态
        # if dsac_agent.use_lstm:
        #     dsac_agent.reset_lstm_states()
        if episode_count >= OVER_EPOCH:
            print("训练结束")
            break

        while not done:
            step_start_time = time.time()  # 记录步骤开始时间


            a_start_time = time.time()
            action_continuous = dsac_agent.choose_action(state)
            a_end_time = time.time() - a_start_time


            # 这里的 dim_discrete_actions 应该使用 env_discrete_action_size
            action_env = map_continuous_to_discrete_better(action_continuous[0], env_discrete_action_size)

            # env.step 的返回值根据您的环境定义
            s_start_time = time.time()
            next_state, reward, done = env.step(state, action_env)
            s_end_time = time.time() - s_start_time

            dsac_agent.store_transition(state, action_continuous, reward, next_state, done)
            buffer_size = len(dsac_agent.memory)

            if buffer_size > batch_size  and global_step % train_freq == 0:
                learn_time = time.time()
                critic_loss, actor_loss = dsac_agent.learn()
                learn_time_end = time.time() - learn_time
                writer.add_scalar("Train/critic_loss", critic_loss, global_step)
                writer.add_scalar("Train/actor_loss", actor_loss, global_step)

                if dsac_agent.auto_alpha:
                    writer.add_scalar("Train/alpha", dsac_agent.log_alpha.exp().item(), global_step)
                else:
                    writer.add_scalar("Train/alpha", dsac_agent.alpha, global_step)
            # 如果buffer_size不足，loss不应该被计算，但可以设定一个默认值
            else:
                learn_time_end = 0
                critic_loss, actor_loss = 0.0, 0.0  # 或者 None, None，取决于后续如何使用

            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1


            # 检查episode_step_limit*
            if episode_length >= episode_step_limit:
                print(f"Epoch {episode_count}, Step {episode_length}: Reached step limit, setting done=True.")
                done = True

            step_end_time = time.time()  # 记录步骤结束时间
            step_times.append(step_end_time - step_start_time)  # 记录本步骤耗时
            a_times.append(a_end_time)
            s_times.append(s_end_time)
            l_times.append(learn_time_end)


        avg_step_time = sum(step_times) / len(step_times) if step_times else 0  # 计算平均步骤时间
        avg_env_step_time = sum(s_times) / len(s_times) if s_times else 0
        avg_a_step_time = sum(a_times) / len(a_times) if a_times else 0
        avg_l_time = sum(l_times) / len(l_times) if l_times else 0
        print(
            f"Episode: {episode_count}, "
            f"Avg Reward: {episode_reward:.2f}, "
            f"Avg Length: {episode_length:.2f}, "
            f"time: {avg_step_time} {avg_env_step_time} {avg_a_step_time} {avg_l_time}")
        # 记录每个 episode 的奖励和长度
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        writer.add_scalar("Episode/reward", episode_reward, episode_count)
        writer.add_scalar("Episode/length", episode_length, episode_count)
        writer.add_scalar("Episode/avg_step_time", avg_step_time, episode_count)
        writer.add_scalar("Episode/avg_env_step_time", avg_env_step_time, episode_count)
        writer.add_scalar("Episode/avg_action_step_time", avg_a_step_time, episode_count)
        # 打印日志和TensorBoard更新的频率一致**
        if episode_count > 0 and episode_count % log_interval == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            # print(
            #     f"Episode: {episode_count}, "
            #     f"Avg Reward: {avg_reward:.2f}, "
            #     f"Avg Length: {avg_length:.2f}, "
            #     f"Alpha: {current_alpha:.4f}, "
            #     f"Step: {global_step}",
            #     f"time: {avg_step_time}")
            current_alpha = dsac_agent.log_alpha.exp().item() if dsac_agent.auto_alpha else dsac_agent.alpha
            writer.add_scalar("Reward/Cumulative_Epoch_Reward", avg_reward,
                              episode_count)  
            writer.add_scalar("Reward/Cumulative_Step_Reward", avg_reward,
                              global_step)  
            writer.add_scalar("Reward/Avg_Length_Last_X_Episodes", avg_length, episode_count)  # 增加平均长度记录

        if episode_count > 0 and episode_count % save_interval == 0:
            dsac_agent.save_model(episode_count, save_dir)

        episode_count += 1

    env.close()
    writer.close()

def main():

    results_path = r"E:/CMVC/Proj/usvlib4ros_origin/results/"
    method_name = f"dsacv2_cnn"  # {datetime.now().strftime('%m%d-%H%M')}"
    log_dir = os.path.join(results_path, method_name)  

    TRAIN_PARAMS = {
        'initial_model_path': None,
        # 例如：r"E:\CMVC\Proj\usvlib4ros_origin\results\dsacv2_some_date\checkpoints\checkpoint_epoch_X.pt"
        # 4600
        'resume_model_path': r"E:\CMVC\Proj\usvlib4ros_origin\results\dsacv2_cnn\checkpoints\dsacv2_checkpoint_epoch_8600.pt", # r"E:\CMVC\Proj\usvlib4ros_origin\results\dsacv2_1\checkpoints\dsacv2_checkpoint_epoch_8000.pt", # None,
        # 例如：r"E:\CMVC\Proj\usvlib4ros_origin\results\dsacv2_some_date\checkpoints\checkpoint_epoch_Y.pt"
        'save_dir': os.path.join(log_dir, "checkpoints"),  
        'log_dir': log_dir, 
        'use_cnn': True,
        "use_lstm": False,
        'dim_states': 97,  # 注意：这个维度需要和您的实际环境状态维度匹配  97->MLP->7     CNN
        'dim_actions': 1,  # DSACv2 连续动作空间维度 (例如，输出5个控制量)
        "total_timesteps": 8_000_000, 
        'tau': 0.005,
        'tau_b': 0.005,
        'actor_lr': 1e-4,
        'critic_lr': 3e-4,
        'alpha_lr': 1e-4,
        'gamma': 0.99,
        'batch_size': 512,
        'memory_capacity': 50000, # 50000, 
        'alpha': 0.2,
        'auto_alpha': True,
        'target_entropy': None,
        "gradient_adjustment_eta": 0.8,
        "lstm_hidden_size": 64,
        "lstm_layers": 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'log_interval': 100,  
        'save_interval': 100,  
        'episode_step_limit': 800,  # 增加episode_step_limit**
    }

    #
    RUN_MODE = "train"  # 默认为训练模式

    print(f"运行模式:{RUN_MODE}")

    if RUN_MODE == "train":
        # *日志目录创建和参数保存
        # 如果不是从检查点继续训练，则创建新的日志目录并保存配置
        if not TRAIN_PARAMS['resume_model_path']:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(os.path.join(log_dir, "config.json"), 'w', encoding='utf-8') as f:
                json.dump(TRAIN_PARAMS, f, ensure_ascii=False, indent=4)
        print(TRAIN_PARAMS)
        train_dsacv2_agent(**TRAIN_PARAMS)
    elif RUN_MODE == "test":
        print("DSACv2 尚未提供测试模式实现。请自行实现 test_dsacv2_agent 函数。")
    else:
        print("请设置 RUN_MODE 为 'train' 或 'test'。")


if __name__ == "__main__":
    main()
