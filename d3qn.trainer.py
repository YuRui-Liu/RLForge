import time
import sys
import json
from collections import deque
import re
sys.path.append(r"E:\\CMVC\\Proj\\usvlib4ros_origin")
import torch
import numpy as np
from algor.d3qn import DQN
import os
from usvlib4ros.usvRosUtil import LogUtil, USVRosbridgeClient
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

MAX_EPOCH = 10000
MAX_EPOCH = 500 # 课程学习

def test_model(
        resume_model_path: str,  # 必须提供的模型检查点路径
        device='cuda' if torch.cuda.is_available() else 'cpu'  # 指定运行设备
):
    """
    从指定模型路径加载模型和配置进行测试。

    Args:
        resume_model_path (str): 训练好的模型检查点文件的完整路径。
        device (str): 运行测试的设备 ('cuda' 或 'cpu')。
    """
    # --- 从模型路径解析配置和日志目录 --- #
    if not resume_model_path:
        raise ValueError("resume_model_path 必须提供用于测试的模型路径。")

    # 假设模型路径格式为: results\METHOD_NAME\checkpoints\checkpoint_epoch_XXX.pt
    # 或者直接是 results\METHOD_NAME\checkpoint_epoch_XXX.pt
    # 我们需要从路径中提取 'METHOD_NAME' 来找到对应的 'config.json'
    # 使用正则表达式匹配 'results\' 后的第一个文件夹名作为 method_name
    match = re.search(r"results[\\/]([A-Za-z0-9@_/-]+)[\\/]", resume_model_path)
    if not match:
        # 如果模型直接在 results/METHOD_NAME 下，再尝试从模型文件的父目录中提取
        # 例如: results\d3qn_0725-2050\checkpoint_epoch_1000.pt
        parent_dir = os.path.dirname(resume_model_path)
        match = re.search(r"[\\/]([A-Za-z0-9@_/-]+)$", parent_dir)
        if not match:
            raise ValueError(f"无法从模型路径解析方法名（例如，d3qn_0725-2050）：{resume_model_path}")

    method_name = match.group(1)  # 提取到方法名
    base_results_dir = r"E:\CMVC\Proj\usvlib4ros_origin\results"  # 结果保存的基目录
    log_dir = os.path.join(base_results_dir, method_name)  # 构建日志目录路径
    config_path = os.path.join(log_dir, "config.json")  # 构建配置文件路径

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"未找到对应的配置文件: {config_path}\n请确保模型路径下的logs文件夹中有config.json文件。")

    # 加载训练时的配置参数
    with open(config_path, 'r', encoding='utf-8') as f:
        loaded_params = json.load(f)

    print(f"从 {config_path} 加载配置参数。")
    print(f"将要加载的模型检查点: {resume_model_path}")

    # --- 环境和智能体初始化 --- #

    # 使用从配置文件加载的dim_actions来初始化环境
    # env = [... 使用自己的环境...]

    # 使用从配置文件加载的参数初始化 DQN 智能体，并加载模型
    # 注意：这里使用 .get(key, default_value) 的方式获取参数，以避免配置文件中缺少某些参数时报错，
    # 并且处理了键名大小写可能不同的情况（例如 'use_C51' vs 'use_c51'）。
    dqn_agent = DQN(
        model_path=resume_model_path,  # 最关键的参数，指定要加载的模型权重
        use_cnn=loaded_params.get('use_cnn', False),
        use_lstm=loaded_params.get('use_lstm', False),
        use_double_dqn=loaded_params.get('use_double_dqn', True),
        use_dueling=loaded_params.get('use_dueling', True),
        use_per=loaded_params.get('use_per', True),
        use_noisy_net=loaded_params.get('use_noisy_net', False),
        use_C51=loaded_params.get('use_c51', False),  # 注意这里的键名，应与config.json中的一致
        use_MSL=loaded_params.get('use_msl', False),  # 注意这里的键名，应与config.json中的一致
        dim_states=loaded_params['dim_states'],
        dim_actions=loaded_params['dim_actions'],
        tau=loaded_params.get('tau', 0.005),
        lr=loaded_params.get('lr', 1e-4),
        gama=loaded_params.get('gamma', 0.99),  # 注意这里的键名
        batch_size=loaded_params.get('batch_size', 512),
        memory_capacity=loaded_params.get('memory_capacity', 50000),
        epsilon_start=loaded_params.get('epsilon_start', 1.0),
        epsilon_end=loaded_params.get('epsilon_end', 0.1),
        epsilon_decay=loaded_params.get('epsilon_decay', 500),
        per_alpha=loaded_params.get('per_alpha', 0.6),
        per_beta_start=loaded_params.get('per_beta_start', 0.4),
        per_beta_end=loaded_params.get('per_beta_end', 1.0),
        per_beta_deacy_steps=loaded_params.get('per_beta_decay_steps', 100000),  # 注意这里的键名
        n_step=loaded_params.get('n_step', 1),
        v_min=loaded_params.get('v_min', -10),
        v_max=loaded_params.get('v_max', 10),
        atoms = loaded_params.get('atoms', 51),
        lstm_hidden_size = loaded_params.get('lstm_hidden_size', 128),
        lstm_layers = loaded_params.get('lstm_layers', 1),
        device = device
    )

    # 初始化测试状态指标
    episode_reward_history = []  # 记录每个回合的总奖励
    episode_length_history = []  # 记录每个回合的步数
    goal_count = 0  # 成功抵达目标点的次数
    # 定义总共测试多少个回合，可以固定，也可以从配置中读取
    total_episodes_to_test = 100  # 例如，测试100个回合

    print("开始测试...")



    # 循环进行多个测试回合
    for episode_idx in range(total_episodes_to_test):
        print(f"\n--- 正在测试回合 {episode_idx + 1}/{total_episodes_to_test} ---")
        state = env.reset()  # 重置环境，开始新回合
        done = False  # 回合是否结束的标志
        episode_reward = 0  # 当前回合的总奖励
        episode_length = 0  # 当前回合的步数

        # 从配置文件获取最大步数限制，如果未设置则默认为800步
        episode_step_limit = loaded_params.get('episode_step_limit', 800)

        # 测试循环，直到回合结束（达到目标，或环境判断为done，或达到步数上限）
        while not done and not env.get_goalbox and episode_length < episode_step_limit:

            # 在测试时，智能体通常不需要探索，直接选择当前最优动作
            action = dqn_agent.choose_action(state, testing=True)

            # env.step 的返回值根据你提供的测试代码进行匹配：next_state, reward, done, max_distance
            next_state, reward, done_env_step = env.step(state, action)

            state = next_state  # 更新当前状态
            # episode_reward += reward  # 累加奖励
            episode_length += 1  # 累加步数
            done = done_env_step  # 更新done状态，如果环境返回了True，则回合结束
            episode_reward += -0.1
            # 更新 ROS 状态，这里主要是为了实时显示测试进度，不影响测试逻辑
            # episode_idx + 1 表示当前是第几个测试回合
            # 如果环境返回 done=True，表示回合因其他原因结束（如撞墙、超出范围等）
            if done:
                if reward > 500:
                    goal_count += 1  # 成功计数加一
                    episode_reward += 100
                    print(f"回合 {episode_idx + 1}: 成功抵达目标！")
                else:
                    episode_reward -= 100
                    print(f"回合 {episode_idx + 1}: 环境终止（可能超出范围或发生碰撞）。")
                break
        # 记录当前回合的结果
        episode_reward_history.append(episode_reward)
        episode_length_history.append(episode_length)

        # 打印当前回合的统计信息
        # 计算当前的平均奖励、平均步数和成功率
        avg_reward_so_far = np.mean(episode_reward_history)
        avg_length_so_far = np.mean(episode_length_history)
        success_rate = goal_count / (episode_idx + 1) if (episode_idx + 1) > 0 else 0

        print(f"回合 {episode_idx + 1} 总结:")
        print(f"  总奖励: {episode_reward:.2f}, 总步数: {episode_length}, 总成功步数: {goal_count}")
        print(f"  累计成功率 (SR): {success_rate:.3f}, 累计平均步数: {avg_length_so_far:.2f}, 累计平均奖励: {avg_reward_so_far:.2f}")

    print("\n--- 测试完成 ---")
    # 计算最终的测试结果
    final_success_rate = goal_count / total_episodes_to_test
    final_avg_reward = np.mean(episode_reward_history)
    final_avg_length = np.mean(episode_length_history)
    print(f"总测试回合数: {total_episodes_to_test}")
    print(f"成功抵达目标次数: {goal_count}")
    print(f"最终成功率 (SR): {final_success_rate:.3f}")
    print(f"最终平均每回合奖励: {final_avg_reward:.2f}")
    print(f"最终平均每回合步数: {final_avg_length:.2f}")


def train_d3qn_agent(
        total_timesteps,
        dim_states,
        dim_actions,
        use_cnn,
        use_lstm,
        use_double_dqn,
        use_dueling,
        use_per,
        use_noisy_net,
        use_c51,
        use_msl,
        tau,
        lr,
        gamma,
        batch_size,
        memory_capacity,
        epsilon_start,
        epsilon_end,
        epsilon_decay,
        per_alpha,
        per_beta_start,
        per_beta_end,
        per_beta_decay_steps,
        n_step,
        v_min,
        v_max,
        atoms,
        lstm_hidden_size,
        lstm_layers,
        save_dir,
        log_dir,
        resume_model_path=None,
        initial_model_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_interval=100,
        save_interval=100,
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
        method_name = re.search(r"results\\([A-Za-z0-9@_/-]+)", resume_model_path).group(1)
        print("加载检查点:", method_name)
        log_dir = os.path.join(r"E:\CMVC\Proj\usvlib4ros_origin\results", method_name)
        save_dir = log_dir + r"\checkpoints"
        print("模型保存位置:", save_dir)
        model_path_type = "resume"


    # --- 环境和智能体初始化 --- #


    # --- 初始化 DQN 智能体 --- #
    dqn_agent = DQN(
        model_path=resume_model_path,
        model_path_type=model_path_type,
        use_cnn=use_cnn,
        use_lstm=use_lstm,
        use_double_dqn=use_double_dqn,
        use_dueling=use_dueling,
        use_per=use_per,
        use_noisy_net=use_noisy_net,
        use_C51=use_c51,
        use_MSL=use_msl,
        dim_states=dim_states,
        dim_actions=dim_actions,
        tau=tau,
        lr=lr,
        gama=gamma,
        batch_size=batch_size,
        memory_capacity=memory_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        per_alpha=per_alpha,
        per_beta_start=per_beta_start,
        per_beta_end=per_beta_end,
        per_beta_deacy_steps=per_beta_decay_steps,
        n_step=n_step,
        v_min=v_min,
        v_max=v_max,
        atoms=atoms,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        device=device
    )


    # 创建日志目录和模型保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    writer = SummaryWriter(log_dir=log_dir)

    # 初始化变量
    episode_rewards = deque(maxlen=MAX_EPOCH)
    episode_lengths = deque(maxlen=MAX_EPOCH)  # log_interval
    global_step = dqn_agent.total_steps
    episode_count = dqn_agent.start_epoch
    start_time = time.time()

    print(f"开始训练，总步数: {total_timesteps}, 使用设备: {device}")

    while global_step < total_timesteps:
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        # print(f"state: {state} \n {len(state)}")
        while not done:
       

            action = dqn_agent.choose_action(state)
            next_state, reward, done = env.step(state, action)
            # done = terminated or truncated

            dqn_agent.store_transition(state, action, reward, next_state, done)
            buffer_size = len(dqn_agent.memory) if not dqn_agent.use_per else dqn_agent.memory.tree.n_entries

            if buffer_size > batch_size:
                loss = dqn_agent.learn()
                writer.add_scalar("Train/train_loss", loss, global_step)
            else:
                loss = 0

            state = next_state
            episode_reward += reward
            episode_length += 1
            global_step += 1

            if not use_noisy_net:
                dqn_agent.decay_epsilon()

            if episode_length >= episode_step_limit:
                print(f"Epoch {episode_count}, Step {episode_length}: Reached step limit, setting done=True.")
                done = True

        # 记录每个 episode 的奖励
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        writer.add_scalar("Episode/reward", episode_reward, episode_count)
        writer.add_scalar("Episode/length", episode_length, episode_count)
        if not use_noisy_net:  # 只有非NoisyNet才记录epsilon
            writer.add_scalar("Train/epsilon", dqn_agent.epsilon, episode_count)

        if episode_count > 0 and episode_count % log_interval == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            print(
                f"Episode: {episode_count}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Epsilon: {dqn_agent.epsilon:.4f}, Step: {global_step}")
            writer.add_scalar("Reward/Cumulative_Epoch_Reward", avg_reward, episode_count)
            writer.add_scalar("Reward/Cumulative_Step_Reward", avg_reward, global_step)
            writer.add_scalar("Reward/length", episode_length, episode_count)

        if episode_count > 0 and episode_count % save_interval == 0:
            dqn_agent.save_model(episode_count, save_dir)

        episode_count += 1


    writer.close()
    print(f"训练结束，耗时: {(time.time() - start_time) / 60:.2f} 分钟")


def main():
    results_path = r"E:/CMVC/Proj/usvlib4ros_origin/results/"
    method_name = f"d3qn_{datetime.now().strftime('%m%d-%H%M')}"
    log_dir = results_path + method_name
    TRAIN_PARAMS = {
        'initial_model_path': None, # r"E:\CMVC\Proj\usvlib4ros_origin\results\d3qn_0726-2041\checkpoints\checkpoint_epoch_2800.pt ",# None,
        'resume_model_path': None, # "E:\CMVC\Proj\usvlib4ros_origin\results\d3qn_0727-1554\checkpoints\checkpoint_epoch_1300.pt",  # r"E:\CMVC\Proj\usvlib4ros_origin\results\d3qn_0725-2050\checkpoints\checkpoint_epoch_1200.pt", # None,  # 设置为模型路径如："runs/d3qn/checkpoints/checkpoint_epoch_100.pt" 用于继续训练
        'save_dir': log_dir + "/checkpoints",
        'log_dir': log_dir,
        'use_cnn': True,  # 如果是图像输入设为 True
        "use_lstm": False,
        'use_double_dqn': True,
        'use_dueling': True,
        'use_noisy_net': True,
        'use_per': True,
        "use_c51": False,
        "use_msl": False,
        'dim_states': 97,  # 状态空间维度
        'dim_actions': 5,  # 动作空间大小
        "total_timesteps": 8_000_000,  # 总训练步数
        'tau': 0.005,  # 软更新目标网络的 tau
        'lr': 1e-4,  # 学习率
        'gamma': 0.99,  # 折扣因子
        'batch_size': 512,  # 每次训练的 batch 大小
        'memory_capacity': 50000,  # 经验回放缓冲区大小
        'epsilon_start': 1.0,  # 初始探索率
        'epsilon_end': 0.1,  # 最小探索率
        'epsilon_decay': 500,  # 探索率衰减速率
        'per_alpha': 0.6,  # Prioritized Experience Replay 的 alpha 参数
        'per_beta_start': 0.4,  # 初始 beta
        'per_beta_end': 1.0,  # 最终 beta
        'per_beta_decay_steps': 100000,  # beta 的衰减步数
        "n_step": 3,
        "v_min": -10,
        "v_max": 10,
        "lstm_hidden_size": 128,
        "lstm_layers": 1,
        "atoms": 51,
        "episode_step_limit": 1e3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
    }
    TEST_MODEL_PATH = r"E:\CMVC\Proj\usvlib4ros_origin\results\d3qn_0728-2206\checkpoints\checkpoint_epoch_5200.pt"
    # --- 选择运行模式 ---
    # RUN_MODE = "test" # 将这里改为 "train" 进行训练，改为 "test" 进行测试
    RUN_MODE = "train"
    print(f"运行模式:{RUN_MODE}")
    if RUN_MODE == "train":
        # 如果不是从检查点继续训练，则创建新的日志目录并保存配置
        if not TRAIN_PARAMS['resume_model_path']:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(log_dir + "/config.json", 'w', encoding='utf-8') as f:
                json.dump(TRAIN_PARAMS, f, ensure_ascii=False, indent=4) # 将训练参数保存到config.json

        # 调用训练函数
        train_d3qn_agent(**TRAIN_PARAMS)
    elif RUN_MODE == "test":
        # 在测试前检查模型路径是否存在
        if not os.path.exists(TEST_MODEL_PATH):
            print(f"错误: 指定的测试模型路径不存在: {TEST_MODEL_PATH}")
            print("请检查 TEST_MODEL_PATH 是否正确，或先进行训练以生成模型。")
            return # 如果模型不存在则退出

        print(f"----- 开始模型测试 (将加载模型: {TEST_MODEL_PATH}) -----")
        # 调用测试函数，只传递模型路径和设备信息
        test_model(resume_model_path=TEST_MODEL_PATH, device=TRAIN_PARAMS['device'])
    else:
        print("请设置 RUN_MODE 为 'train' 或 'test'。")

if __name__ == "__main__":
    main()
# tensorboard --logdir="E:\CMVC\Proj\usvlib4ros_origin\results"