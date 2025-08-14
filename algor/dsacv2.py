from copy import deepcopy
import random
import torch.nn.functional as F
import math
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from collections import deque


def map_continuous_to_discrete_better(action_continuous, dim_discrete_actions):
    """
    更合理的连续到离散动作映射
    """
    if dim_discrete_actions <= 0:
        # 连续动作空间
        return action_continuous

    elif dim_discrete_actions == 1:
        # 特殊情况：只有一个动作
        return 0
    elif dim_discrete_actions == 2:
        # 二元动作：负数->0, 非负数->1
        return 0 if action_continuous < 0 else 1

    else:
        # 多元动作：将连续值映射到离散动作空间
        # 使用tanh将连续值压缩到[-1, 1]，然后映射到[0, dim_discrete_actions-1]
        # normalized = (np.tanh(action_continuous) + 1) / 2  # 映射到[0, 1]
        normalized = (action_continuous + 1) / 2
        action_env = int(normalized * dim_discrete_actions)
        # 确保不超出范围
        action_env = min(action_env, dim_discrete_actions - 1)
        return action_env



class Actor(nn.Module):
    def __init__(self, dim_states, dim_actions, use_cnn, cnn_tar_dim, use_lstm, lstm_hidden_size, lstm_layers):
        super().__init__()
        self.dim_actions = dim_actions
        self.use_cnn = use_cnn
        self.cnn_tar_dim = cnn_tar_dim 
        self.use_lstm = use_lstm 
        self.lstm_hidden_size = lstm_hidden_size 
        self.lstm_layers = lstm_layers 

        current_feature_dim = dim_states  # 初始特征维度
        hidden_size = 256 # 128 # 256
        # CNN 网络构建
        if self.use_cnn:
            if cnn_tar_dim >= dim_states:
                raise ValueError("cnn_tar_dim must be less than dim_states for splitting.")
            if cnn_tar_dim <= 0:
                raise ValueError("cnn_tar_dim must be positive for CNN processing.")

            # CNN 模块设计
            # Conv1d 输入: (batch_size, channels, length) -> 这里 channels=1, length=cnn_tar_dim
            # Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
            # MaxPool1d(kernel_size=2)
            # 对于 cnn_tar_dim=7, Conv1d 之后长度仍为 7, MaxPool1d 之后长度变为 floor((7-2)/2)+1 = 3

            cnn_output_channels = 32
            # 计算 MaxPool1d 后的特征长度
            # MaxPool1d: L_out = floor((L_in - kernel_size) / stride) + 1
            # 这里 L_in 是 cnn_tar_dim (经过 Conv1d 且 padding=1 后长度不变), kernel_size=2, stride=2 (默认)
            cnn_intermediate_length = int(np.floor((self.cnn_tar_dim - 2) / 2) + 1)

            if cnn_intermediate_length <= 0:
                # 如果 cnn_tar_dim 太小导致 MaxPool1d 无法有效工作，可能需要调整 CNN 结构
                print(f"Warning: cnn_tar_dim={self.cnn_tar_dim} is too small for MaxPool1d(2). Consider adjusting CNN architecture or cnn_tar_dim.")
                # 这里我们保持原设计，假定 cnn_tar_dim 足够大（例如 7 已经足够）

            self.cnn_block = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=cnn_output_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(cnn_output_channels),  # 稳定训练，加速收敛
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),  # 缩小特征长度
                nn.Flatten(),  # 展平为一维向量
                nn.Dropout(p=0.2),
                # 最后的线性层将展平的 CNN 特征压缩到 (dim_states - cnn_tar_dim) 维度
                nn.Linear(cnn_output_channels * cnn_intermediate_length, dim_states - cnn_tar_dim),
                nn.ReLU()
            )
            # 更新特征维度：CNN 处理后的 (dim_states - cnn_tar_dim) 维度 + 未处理的 (dim_states - cnn_tar_dim) 维度
            current_feature_dim = (dim_states - cnn_tar_dim) + (dim_states - cnn_tar_dim)

        # LSTM 网络构建
        # if self.use_lstm:
        #     # LSTM 输入为当前计算出的特征维度 (可能是原始 dim_states，或 CNN 处理后的维度)
        #     self.lstm_block = nn.LSTM(
        #         input_size=current_feature_dim,
        #         hidden_size=self.lstm_hidden_size,
        #         num_layers=self.lstm_layers,
        #         batch_first=True  # (batch, seq_len, features)
        #     )
        #     # LSTM 输出为隐藏层大小
        #     current_feature_dim = self.lstm_hidden_size

        # 主全连接层 (FC) 的输入维度根据 CNN 和 LSTM 的使用情况动态调整
        self.fc = nn.Sequential(
            nn.Linear(current_feature_dim, hidden_size),  # 输入维度是 current_feature_dim
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_actions * 2)  # dim_actions * 2 for mean and log_std
        )

    def forward(self, x):
        processed_x = x  # 初始化处理后的输入为原始输入

        # CNN 处理逻辑
        if self.use_cnn:
            # 拆分状态：前 cnn_tar_dim 部分用于 CNN，其余部分保留
            x_cnn = x[:, :self.cnn_tar_dim]  # 取前 cnn_tar_dim 维度
            x_rest = x[:, self.cnn_tar_dim:]  # 取剩余维度

            # Conv1d 需要 (batch_size, channels, length) 格式
            x_cnn = x_cnn.unsqueeze(1)  # 添加通道维度 (1)

            x_cnn_processed = self.cnn_block(x_cnn)

            # 将 CNN 处理后的特征与未处理的特征在最后一维拼接
            processed_x = torch.cat([x_cnn_processed, x_rest], dim=-1)

        # LSTM 处理逻辑
        # if self.use_lstm:
        #     # LSTM 输入需要 (batch_size, sequence_length, input_size) 格式
        #     # 对于单个时间步的状态，sequence_length = 1
        #     processed_x = processed_x.unsqueeze(1)  # 添加 sequence_length 维度
        #
        #     # LSTM 返回 (output, (h_n, c_n))，我们只取 output
        #     # output: (batch_size, sequence_length, hidden_size)
        #     processed_x, _ = self.lstm_block(processed_x)
        #
        #     # 移除 sequence_length 维度，以便输入到 FC 层
        #     processed_x = processed_x.squeeze(1)

        return self.fc(processed_x)

    def get_act_dist(self, logits):
        """
        根据 logits 创建动作分布。
        logits 预期包含均值和对数标准差。
        """
        logits_mean, logits_log_std = torch.chunk(logits, chunks=2, dim=-1)
        # 限制对数标准差的范围以保证数值稳定性
        logits_log_std = torch.clamp(logits_log_std, -20, 2)
        std = torch.exp(logits_log_std)
        return ReparamTanhNormal(logits_mean, std)

    def get_action(self, obs: torch.Tensor):
        """
        根据观测值获取动作。
        Args:
            obs (torch.Tensor): 当前的状态观测。

        Returns:
            torch.Tensor: 采样得到的动作。
            torch.Tensor: 动作的对数概率（如果 deterministic 为 True，则返回 None 或一个指示值）。
        """
        with torch.no_grad():
            # 1. 通过 Actor 的前向传播获取均值和对数标准差的logits
            logits = self.forward(obs)
            # 2. 从 logits 创建动作分布
            act_dist = self.get_act_dist(logits)
            action = torch.tanh(act_dist.loc)

            # 获取动作的对数概率
            log_prob = act_dist.log_prob(action)
        if self.dim_actions == 1:
            # 返回形状为[batch_size, 1]的tensor，而不是标量
            if action.dim() == 1:
                action = action.unsqueeze(-1)
            elif action.dim() == 0:
                action = action.unsqueeze(0).unsqueeze(-1)

        return action

# DistributionalCritic 网络
# 从状态和动作中输出 Q 值的均值和标准差
class DistributionalCritic(nn.Module):
    def __init__(self, dim_states, dim_actions, use_cnn, cnn_tar_dim, use_lstm, lstm_hidden_size, lstm_layers):
        super().__init__()
        self.use_cnn = use_cnn
        self.cnn_tar_dim = cnn_tar_dim 
        self.use_lstm = use_lstm  
        self.lstm_hidden_size = lstm_hidden_size  
        self.lstm_layers = lstm_layers  

        current_obs_feature_dim = dim_states  # 初始观测特征维度

        # CNN 网络构建
        if self.use_cnn:
            if cnn_tar_dim >= dim_states:
                raise ValueError("cnn_tar_dim must be less than dim_states for splitting.")
            if cnn_tar_dim <= 0:
                raise ValueError("cnn_tar_dim must be positive for CNN processing.")

            cnn_output_channels = 32
            cnn_intermediate_length = int(np.floor((self.cnn_tar_dim - 2) / 2) + 1)

            self.cnn_block = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=cnn_output_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(cnn_output_channels),  # 稳定训练，加速收敛
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(cnn_output_channels * cnn_intermediate_length, dim_states - cnn_tar_dim),
                nn.ReLU()
            )
            current_obs_feature_dim = (dim_states - cnn_tar_dim) + (dim_states - cnn_tar_dim)

        # LSTM 网络构建
        # if self.use_lstm:
        #     self.lstm_block = nn.LSTM(
        #         input_size=current_obs_feature_dim,
        #         hidden_size=self.lstm_hidden_size,
        #         num_layers=self.lstm_layers,
        #         batch_first=True
        #     )
        #     current_obs_feature_dim = self.lstm_hidden_size

        # 主全连接层，输入维度根据 CNN 和 LSTM 的使用情况动态调整
        # Critic 的 FC 输入是处理后的 obs 维度 + 动作维度
        self.fc = nn.Sequential(
            nn.Linear(current_obs_feature_dim + dim_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Outputs mean and std
        )

    def forward(self, obs, act):
        processed_obs = obs  # 原始观测输入

        # CNN 处理逻辑 (与 Actor 类似)
        if self.use_cnn:
            obs_cnn = obs[:, :self.cnn_tar_dim]
            obs_rest = obs[:, self.cnn_tar_dim:]

            obs_cnn = obs_cnn.unsqueeze(1)

            obs_cnn_processed = self.cnn_block(obs_cnn)

            processed_obs = torch.cat([obs_cnn_processed, obs_rest], dim=-1)

        # LSTM 处理逻辑 (与 Actor 类似)
        # if self.use_lstm:
        #     processed_obs = processed_obs.unsqueeze(1)
        #     processed_obs, _ = self.lstm_block(processed_obs)
        #     processed_obs = processed_obs.squeeze(1)

        # 将处理后的观测与动作拼接 (在处理观测完成后，再与动作拼接)
        x = torch.cat([processed_obs, act], dim=-1)
        return self.fc(x)

    def evaluate(self, obs, act):
        """用于 Q 值评估的方法，返回均值和标准差"""
        return self.forward(obs, act)


# 重参数化 Tanh 正态分布
class ReparamTanhNormal:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.normal = Normal(loc, scale)

    def rsample(self):
        x = self.normal.rsample()
        action = torch.tanh(x)
        # 计算 Tanh 变换后的分布的对数概率
        # log_prob = normal_log_prob - sum(log(1 - tanh(x)^2))
        log_prob = self.normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

    def log_prob(self, action):
        epsilon = 1e-6
        action = torch.clamp(action, -1 + epsilon, 1 - epsilon)
        inv_tanh_action = torch.atanh(action)
        log_prob = self.normal.log_prob(inv_tanh_action) - torch.log(1 - action.pow(2) + epsilon).sum(dim=-1,
                                                                                                      keepdim=True)
        return log_prob


# Huber 损失函数，与 F.smooth_l1_loss 兼容
# 参考代码使用 huber_loss(delta=50)，这对应 F.smooth_l1_loss 的 beta 参数
def huber_loss(input, target, delta=1.0, reduction='mean'):
    return F.smooth_l1_loss(input, target, beta=delta, reduction=reduction)


# =================================================================================
# DSAC-T (DSAC-V2) 主类 - 完全按照论文实现
# =================================================================================
class DSACv2:
    def __init__(self, model_path=None,
                 use_cnn=False, use_lstm=False,
                 dim_states=184, dim_actions=5,
                 tau=0.005, tau_b=0.005,  # tau_b 用于 VBGA 的移动平均
                 actor_lr=1e-4, critic_lr=3e-4, alpha_lr=1e-4,
                 gamma=0.99, batch_size=128, memory_capacity=2048,
                 alpha=0.2, auto_alpha=True, target_entropy=None,
                 delay_update=1,  # DSACv2 特有：Actor 网络的延迟更新步数，1 表示不延迟

                 cnn_tar_dim=90,
                 lstm_hidden_size=128, lstm_layers=1,  # （未在学习过程中使用，需要自行添加）
                 device='cpu'
                 ):
        """DSAC-T (DSAC-V2) 初始化"""

        # 保存超参数
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.use_cnn = use_cnn
        self.cnn_tar_dim = cnn_tar_dim  
        self.use_lstm = use_lstm
        self.tau = tau
        self.tau_b = tau_b
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_capacity = int(memory_capacity)
        self.alpha = alpha
        self.auto_alpha = auto_alpha
        self.delay_update = delay_update  

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 设置目标熵
        if target_entropy is None:
            self.target_entropy = -dim_actions  # 默认为 -dim(A)
        else:
            self.target_entropy = target_entropy

        print(f"--- DSAC-T (DSAC-V2) Agent Configuration ---")
        print(f"State dim: {dim_states}, Action dim: {dim_actions}")
        print(f"Using device: {self.device}")
        print(f"Using CNN: {use_cnn}, Using LSTM: {use_lstm}")

        # VBGA 相关参数 - 按照参考代码初始化为 -1.0
        self.mean_std1 = -1.0  # critic 1 的移动平均标准差
        self.mean_std2 = -1.0  # critic 2 的移动平均标准差

        # 经验回放缓冲区
        self.memory = deque(maxlen=self.memory_capacity)
        self.total_steps = 0
        self.start_epoch = 0

        # LSTM 状态（如果使用）
        if self.use_lstm:
            self.hidden_state = None
            self.cell_state = None
        # 网络初始化 - 包含策略网络和双 critic 网络
        self._init_networks()
        # 优化器初始化
        self._init_optimizers()
        # 尝试加载预训练模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("Initialized new networks")

    def choose_action(self, state, testing=False):
        """选择动作 - 支持训练和测试模式"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            # 策略网络输出 logits
            logits = self.actor(state)
            act_dist = self.actor.get_act_dist(logits)  # 调用 Actor 自身的 get_act_dist 方法
            if testing:
                # 测试模式：直接使用 pre-tanh 正态分布的均值
                action = torch.tanh(act_dist.loc)
            else:
                # 训练模式：重参数化采样
                action, _ = act_dist.rsample()
        self.actor.train()
        # 确保即使dim_actions=1也返回正确的格式
        if self.dim_actions == 1:
            # 返回形状为[batch_size, 1]的tensor，而不是标量
            if action.dim() == 1:
                action = action.unsqueeze(-1)
            elif action.dim() == 0:
                action = action.unsqueeze(0).unsqueeze(-1)

            # 返回numpy数组或tensor，保持维度信息
            return action.cpu().numpy()[0]
        else:
            # 多维动作情况
            return action.cpu().numpy()[0]

        # print(action)
        # action.cpu().numpy()[0]

    # 该方法在 Actor.get_act_dist 存在后已不直接使用，但为了兼容性保留
    def create_action_distributions(self, logits_mean, logits_std):
        """创建重参数化 Tanh 正态分布"""
        log_std = torch.clamp(logits_std, -20, 2)
        std = torch.exp(log_std)
        return ReparamTanhNormal(logits_mean, std)

    def store_transition(self, s, a, r, s_, done):
        """存储经验到回放缓冲区"""
        self.memory.append((s, a, r, s_, done))

    def q_evaluate(self, obs, act, qnet):
        """评估 Q 值分布的均值和标准差 - DSAC-T 核心"""
        # critic 网络输出均值和标准差
        StochaQ = qnet.evaluate(obs, act)
        # print(f"StochaQ: {StochaQ.shape}")
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        std = torch.clamp(std, min=0.)  # 确保标准差非负

        # 重参数化采样 - 用于计算具体 Q 值样本
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)  # 截断以保证稳定性
        q_value = mean + torch.mul(z, std)  # 重参数化技巧
        # 确保输出维度正确
        if mean.dim() == 1:
            mean = mean.unsqueeze(-1)  # [128] -> [128, 1]
        if std.dim() == 1:
            std = std.unsqueeze(-1)  # [128] -> [128, 1]
        if q_value.dim() == 1:
            q_value = q_value.unsqueeze(-1)  # [128] -> [128, 1]
        # print(f"mean: {mean.shape}  std: {std.shape}  q_value: {q_value.shape}")
        return mean, std, q_value

    def compute_target_q(self, r, done, q, q_std_avg, q_next, q_next_sample, log_prob_a_next, alpha):
        """计算目标 Q 值 - 实现 Expected Value Substituting (EVS)"""
        # 标准 Bellman 目标 Q 值
        target_q = r + (1 - done) * self.gamma * (q_next - alpha * log_prob_a_next)

        # 采样目标 Q 值 - 用于边界计算
        target_q_sample = r + (1 - done) * self.gamma * (q_next_sample - alpha * log_prob_a_next)

        # Expected Value Substituting: 基于方差的边界裁剪
        # 这里 q_std_avg 是传入的移动平均标准差 (self.mean_std1/2)
        td_bound = 3 * q_std_avg

        difference = torch.clamp(target_q_sample - q, -td_bound, td_bound)  # 裁剪差异
        target_q_bound = q + difference  # 计算边界目标值

        return target_q.detach(), target_q_bound.detach()

    def learn(self):
        """学习步骤 - 实现 DSAC-T 的三个 refinement"""
        self.total_steps += 1  # 累加总步数作为迭代计数器

        # 检查是否有足够的经验进行学习
        if len(self.memory) < self.batch_size:
            return 0, 0  # 如果数据不足，返回 0 损失

        # 采样批次数据
        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_batch)

        b_s = torch.FloatTensor(np.array(states)).to(self.device)
        b_a = torch.FloatTensor(np.array(actions)).to(self.device)
        b_r = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        b_s_ = torch.FloatTensor(np.array(next_states)).to(self.device)
        b_d = torch.FloatTensor(dones).view(-1, 1).to(self.device)
        # print(f"s: {b_s.shape} a:{b_a.shape}  r:{b_r.shape}  s_:{b_s_.shape}  d:{b_d.shape}")
        # 获取当前 alpha 值
        alpha = self.get_alpha(requires_grad=True) if self.auto_alpha else torch.tensor(self.alpha, device=self.device)

        # ================== 1. 更新 Critic 网络 ==================
        # 在计算 target Q 之前清零两个 critic 的梯度
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()

        # ====将所有 target 计算都放在 no_grad 块中 =====

        # 通过目标策略网络采样下一个动作
        logits_2 = self.actor_target(b_s_)
        act2_dist = self.actor_target.get_act_dist(logits_2)  # 调用 Actor 的方法
        act2, log_prob_act2 = act2_dist.rsample()

        # 评估当前状态-动作对的 Q 值（用于损失计算）
        q1_current, q1_std_current, _ = self.q_evaluate(b_s, b_a, self.critic_1)
        q2_current, q2_std_current, _ = self.q_evaluate(b_s, b_a, self.critic_2)

        # 更新 VBGA 的移动平均标准差 (mean_std1/2)
        # 它们是标量 Tensor，不参与梯度计算，仅作为统计量
        if isinstance(self.mean_std1, float) and self.mean_std1 == -1.0:  # 首次更新时，从 float 变为 Tensor
            self.mean_std1 = torch.mean(q1_std_current.detach())
        else:
            self.mean_std1 = (1 - self.tau_b) * self.mean_std1 + self.tau_b * torch.mean(q1_std_current.detach())
        self.mean_std1 = self.mean_std1

        if isinstance(self.mean_std2, float) and self.mean_std2 == -1.0:  # 首次更新时，从 float 变为 Tensor
            self.mean_std2 = torch.mean(q2_std_current.detach())
        else:
            self.mean_std2 = (1 - self.tau_b) * self.mean_std2 + self.tau_b * torch.mean(q2_std_current.detach())
        self.mean_std2 = self.mean_std2

        with torch.no_grad():
            # 评估下一个状态的 Q 值（用于目标 Q 计算）
            q1_next, _, q1_next_sample = self.q_evaluate(b_s_, act2, self.critic_1_target)
            q2_next, _, q2_next_sample = self.q_evaluate(b_s_, act2, self.critic_2_target)

            # 使用两个目标 Q 值的最小值
            q_next_min = torch.min(q1_next, q2_next)
            q_next_sample_min = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

            # 计算目标 Q 值
            # 将 mean_std1/2 作为 q_std_avg 传入
            target_q1, target_q1_bound = self.compute_target_q(
                b_r, b_d, q1_current,
                self.mean_std1,  # 传入移动平均标准差
                q_next_min, q_next_sample_min, log_prob_act2, alpha
            )

            target_q2, target_q2_bound = self.compute_target_q(
                b_r, b_d, q2_current,
                self.mean_std2,  # 传入移动平均标准差
                q_next_min, q_next_sample_min, log_prob_act2, alpha
            )

        # 计算 critic 1 损失
        q1_std_clamp = torch.clamp(q1_std_current, min=0.).detach()
        bias = 0.1

        ratio1 = (torch.pow(self.mean_std1, 2) / (torch.pow(q1_std_clamp, 2) + bias)).clamp(min=0.1, max=10)
        # print(f"q1_current:{q1_current.shape}  target_q1:{target_q1.shape}  q1_next:{q1_next.shape}  q1_next_sample:{q1_next_sample.shape}")
        # 确保梯度正确传播
        q1_loss_main = huber_loss(q1_current, target_q1, delta=50, reduction='none')
        q1_loss_std_part1 = q1_std_clamp.pow(2)
        q1_loss_std_part2 = huber_loss(q1_current.detach(), target_q1_bound, delta=50, reduction='none')
        q1_loss_std = q1_std_current * (q1_loss_std_part1 - q1_loss_std_part2) / (q1_std_clamp + bias)
        critic_1_loss = torch.mean(ratio1 * (q1_loss_main + q1_loss_std))

        # 计算 critic 2 损失
        q2_std_clamp = torch.clamp(q2_std_current, min=0.).detach()

        ratio2 = (torch.pow(self.mean_std2, 2) / (torch.pow(q2_std_clamp, 2) + bias)).clamp(min=0.1, max=10)

        q2_loss_main = huber_loss(q2_current, target_q2, delta=50, reduction='none')
        q2_loss_std_part1 = q2_std_clamp.pow(2)
        q2_loss_std_part2 = huber_loss(q2_current.detach(), target_q2_bound, delta=50, reduction='none')
        q2_loss_std = q2_std_current * (q2_loss_std_part1 - q2_loss_std_part2) / (q2_std_clamp + bias)
        critic_2_loss = torch.mean(ratio2 * (q2_loss_main + q2_loss_std))

        # 更新 Critic 1 网络
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # 更新 Critic 2 网络
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # ================== 2. 更新 Actor 网络 (延迟更新) ==================
        # 仅当总步数是 delay_update 的倍数时才更新 Actor
        if self.total_steps % self.delay_update == 0:
            # 冻结 critic 网络梯度，避免 actor 的梯度回传到 critic
            for p in self.critic_1.parameters():
                p.requires_grad = False
            for p in self.critic_2.parameters():
                p.requires_grad = False

            # 策略更新
            logits = self.actor(b_s)
            act_dist = self.actor.get_act_dist(logits)  # 调用 Actor 的方法
            new_act, new_log_prob = act_dist.rsample()

            # 评估新动作的 Q 值
            q1_new, _, _ = self.q_evaluate(b_s, new_act, self.critic_1)
            q2_new, _, _ = self.q_evaluate(b_s, new_act, self.critic_2)

            # 双 Q 网络学习 (Twin Value Distribution Learning)
            min_q_new = torch.min(q1_new, q2_new)

            # Actor 损失
            actor_loss = (alpha * new_log_prob - min_q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 恢复 critic 网络梯度
            for p in self.critic_1.parameters():
                p.requires_grad = True
            for p in self.critic_2.parameters():
                p.requires_grad = True

            # ================== 3. 更新 Alpha (如果自动调整) ==================
            if self.auto_alpha:
                self.alpha_optimizer.zero_grad()
                # 损失函数: -log_alpha * (熵 + 目标熵)
                loss_alpha = (-self.log_alpha * (new_log_prob.detach() + self.target_entropy)).mean()
                loss_alpha.backward()
                self.alpha_optimizer.step()

            # ================== 4. 软更新目标网络 (延迟更新) ==================
            self.soft_update(self.critic_1_target, self.critic_1)
            self.soft_update(self.critic_2_target, self.critic_2)
            self.soft_update(self.actor_target, self.actor)
        else:
            # 如果 Actor 没有更新，Actor 损失在此步概念上为 0
            actor_loss = torch.tensor(0.0)

        return (critic_1_loss.item() + critic_2_loss.item()) / 2, actor_loss.item()

    def soft_update(self, target_net, eval_net):
        """软更新目标网络参数"""
        with torch.no_grad():
            polyak = 1 - self.tau  
            for p, p_targ in zip(eval_net.parameters(), target_net.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_alpha(self, requires_grad=False):
        """获取 alpha 值"""
        # 添加 get_alpha 方法 
        if self.auto_alpha:
            alpha = self.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                return alpha.item()
        else:
            return self.alpha

    def save_model(self, epoch, save_dir):
        """保存模型"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"dsacv2_checkpoint_epoch_{epoch}.pt")

        state = {
            'epoch': epoch,
            'total_steps': self.total_steps,
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'critic_1_target_state_dict': self.critic_1_target.state_dict(),
            'critic_2_target_state_dict': self.critic_2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
            'mean_std1': self.mean_std1,  # 这些是 Tensor/float，直接保存值
            'mean_std2': self.mean_std2
        }

        if self.auto_alpha:
            state['log_alpha'] = self.log_alpha  # log_alpha 是 nn.Parameter，直接保存
            state['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()

        torch.save(state, filename)
        print(f"DSAC-V2 Checkpoint saved to {filename}")

    def load_model(self, path):
        """加载模型"""
        print(f"Loading DSAC-V2 checkpoint from: {path}")
        checkpoint = torch.load(path) # , map_location=self.device
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
        self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target_state_dict'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target_state_dict'])

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
        self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])

        # 加载 mean_std 值
        if 'mean_std1' in checkpoint:
            self.mean_std1 = checkpoint['mean_std1']
        if 'mean_std2' in checkpoint:
            self.mean_std2 = checkpoint['mean_std2']

        if self.auto_alpha and 'log_alpha' in checkpoint:
            # 重新初始化 log_alpha 为 nn.Parameter
            if isinstance(checkpoint['log_alpha'], nn.Parameter):
                self.log_alpha = checkpoint['log_alpha'].to(self.device)
            else:  # 假定它是一个 Tensor
                self.log_alpha = nn.Parameter(checkpoint['log_alpha'].to(self.device))
            # 用加载的 log_alpha 参数重新初始化 alpha_optimizer
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.total_steps = checkpoint.get('total_steps', 0)
        print(f"DSAC-V2 Checkpoint loaded. Resuming training from Epoch {self.start_epoch}.")


    def _init_networks(self):
        """初始化网络"""
        self.actor = Actor(self.dim_states, self.dim_actions,
                           self.use_cnn, self.cnn_tar_dim,
                           self.use_lstm, self.lstm_hidden_size, self.lstm_layers).to(self.device)
        self.actor_target = deepcopy(self.actor) # 使用 deepcopy 进行目标网络初始化

        self.critic_1 = DistributionalCritic(self.dim_states, self.dim_actions,
                                             self.use_cnn, self.cnn_tar_dim,
                                             self.use_lstm, self.lstm_hidden_size, self.lstm_layers).to(self.device)
        self.critic_2 = DistributionalCritic(self.dim_states, self.dim_actions,
                                             self.use_cnn, self.cnn_tar_dim,
                                             self.use_lstm, self.lstm_hidden_size, self.lstm_layers).to(self.device)
        self.critic_1_target = deepcopy(self.critic_1)  # 使用 deepcopy 进行目标网络初始化
        self.critic_2_target = deepcopy(self.critic_2)  # 使用 deepcopy 进行目标网络初始化

        # 冻结目标网络梯度
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_1_target.parameters():
            p.requires_grad = False
        for p in self.critic_2_target.parameters():
            p.requires_grad = False

    def _init_optimizers(self):
        """初始化优化器"""
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # 熵温度系数优化器
        if self.auto_alpha:
            # log_alpha 应该是一个 nn.Parameter，与参考代码一致
            self.log_alpha = nn.Parameter(torch.tensor(math.log(self.alpha), dtype=torch.float32, device=self.device))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)


