import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import collections
import json
from pathlib import Path


def set_seeds(seed_value):
    """
    为所有相关的随机数生成器设置种子。
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU
    print(f"[INFO] All random seeds set to: {seed_value}")


# =================================================================================
# 1. 优先经验回放 (PER) 的核心：SumTree 数据结构
# =================================================================================
class SumTree:
    """SumTree数据结构，用于高效地按优先级采样。"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(idx, p)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区。"""

    def __init__(self, capacity, alpha=0.6,device="cpu"):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 0.01
        self.device = device

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        max_p = 1.0 if max_p == 0 else max_p
        self.tree.add(max_p, transition)

    def sample(self, batch_size, beta=0.4):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.n_entries = self.tree.n_entries

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        return batch, np.array(idxs), torch.FloatTensor(is_weights).to(self.device)

    def update_priorities(self, batch_indices, td_errors):
        td_errors = td_errors.detach().cpu().numpy()
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, p in zip(batch_indices, priorities):
            self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


# =================================================================================
# 2. Noisy Nets 的核心：NoisyLinear 层
# =================================================================================
class NoisyLinear(nn.Module):
    """Noisy Net的线性层，用可学习的噪声替代Epsilon-Greedy探索。"""

    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features, self.out_features, self.std_init = in_features, out_features, std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


# =================================================================================
# 3. 融合了 CNN, Dueling, Noisy Nets 的统一网络结构
# =================================================================================
class FusedNet(nn.Module):
    """融合了CNN、Dueling和Noisy Net的可配置网络结构。"""

    def __init__(self, n_states, n_actions, use_cnn=False, use_lstm=False, use_dueling=False, use_noisy=False, use_C51=False, 
                lstm_hidden_size=128, lstm_layers=1,atoms=51,cnn_input_dim=90):
        super(FusedNet, self).__init__()
        self.use_cnn, self.use_lstm, self.use_dueling, self.use_noisy, self.use_C51 = use_cnn, use_lstm, use_dueling, use_noisy, use_C51
        self.n_actions = n_actions
        self.atoms = atoms  # C51的原子数
        self.cnn_input_dim = cnn_input_dim
        linear_layer = NoisyLinear if use_noisy else nn.Linear

        if self.use_cnn:
            self.cnn_base = nn.Sequential(
                nn.Conv1d(1, 16, 8, 4), nn.ReLU(),
                nn.Conv1d(16, 32, 5, 2), nn.ReLU(),
                nn.Conv1d(32, 32, 3, 1), nn.ReLU()
            )
            cnn_output_dim = 32 * 7 # 根据Conv结构计算得出
            # 非CNN部分的维度
            non_cnn_dim = n_states - cnn_input_dim
            feature_dim = cnn_output_dim + non_cnn_dim  # 拼接CNN输出和非CNN特征
        else:
            feature_dim = n_states




        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=0.1 if lstm_layers > 1 else 0
            )
            # LSTM输出维度作为后续层的输入
            feature_dim = lstm_hidden_size

        self.fc_shared = nn.Sequential(
            linear_layer(feature_dim, 128), nn.ReLU(),
            linear_layer(128, 128), nn.ReLU(),
            linear_layer(128, 128), nn.ReLU()  
        )
        if self.use_C51:
            output_size = n_actions * atoms
            if self.use_dueling:
                self.value_stream = linear_layer(128, atoms)
                self.advantage_stream = linear_layer(128, n_actions * atoms)
            else:
                self.out = linear_layer(128, output_size)
        else:
            if self.use_dueling:
                self.value_stream = linear_layer(128, 1)
                self.advantage_stream = linear_layer(128, self.n_actions)
            else:
                self.out = linear_layer(128, self.n_actions)
        # LSTM隐藏状态缓存
        self.hidden_states = None


    def forward(self, x):
        batch_size = x.size(0)
        cnn_out = None
        non_cnn_part = None
        if self.use_cnn:
            # 拆分输入为CNN部分和非CNN部分
            cnn_input = x[:, :self.cnn_input_dim]      # 提取前cnn_input_dim维度用于CNN
            non_cnn_part = x[:, self.cnn_input_dim:]   # 其余维度保留
            # 扩展维度以适配Conv1d [B, channels=1, length]
            if len(cnn_input.shape) == 2:
                cnn_input = cnn_input.unsqueeze(1)     # [B, 1, cnn_input_dim]

            cnn_out = self.cnn_base(cnn_input)
            cnn_out = cnn_out.view(batch_size, -1)     # [B, cnn_output_dim]
        else:
            non_cnn_part = x

        # 拼接CNN输出和非CNN部分
        if cnn_out is not None and non_cnn_part is not None:
            x = torch.cat([cnn_out, non_cnn_part], dim=1)  # 拼接两部分
        elif cnn_out is not None:
            x = cnn_out
        else:
            x = non_cnn_part

            #  LSTM处理
        if self.use_lstm:
            # 重塑为LSTM输入格式 [batch_size, seq_len=1, features]
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # [B, 1, features]
            
            # LSTM前向传播
            lstm_out, self.hidden_states = self.lstm(x, self.hidden_states)
            # 取最后一个时间步的输出
            x = lstm_out[:, -1, :]  # [B, hidden_size]
        else:
            # 如果不使用LSTM，确保维度正确
            if len(x.shape) == 3 and x.shape[1] == 1:
                x = x.squeeze(1)  # [B, features]


        x = self.fc_shared(x)
        if self.use_C51:  # C51输出概率分布
            
            if self.use_dueling: 
                value = self.value_stream(x).view(-1, 1, self.atoms)
                advantage = self.advantage_stream(x).view(-1, self.n_actions, self.atoms)
                q_atoms = value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                q_atoms = self.out(x).view(-1, self.n_actions, self.atoms)
            return F.softmax(q_atoms, dim=-1)  # 返回概率分布
        else:   # 传统DQN输出Q值
            if self.use_dueling:
                value = self.value_stream(x)
                advantage = self.advantage_stream(x)
                q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
            else:
                q_vals = self.out(x)
            return q_vals


    def reset_noise(self):
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def reset_hidden_states(self):
        """重置LSTM隐藏状态"""
        # 添加重置隐藏状态的方法
        if self.use_lstm:
            self.hidden_states = None
    
    def detach_hidden_states(self):
        """分离LSTM隐藏状态以防止梯度回传"""
        # 添加分离隐藏状态的方法
        if self.use_lstm and self.hidden_states is not None:
            h, c = self.hidden_states
            self.hidden_states = (h.detach(), c.detach())
# =================================================================================
# 4. N-step Learning 缓冲区
# =================================================================================
class NStepBuffer:
    """N-step学习的缓冲区，用于计算多步回报"""
    
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = collections.deque(maxlen=n_step)

    def push(self, transition):
        """添加转换并返回n-step转换"""
        self.buffer.append(transition)
        if len(self.buffer) < self.n_step:
            return None
        
        # 计算n-step回报
        R = 0
        for i in reversed(range(self.n_step)):
            R = self.buffer[i][2] + self.gamma * R  # reward
        
        # 获取初始状态和动作，最终状态和完成标志
        s, a = self.buffer[0][0], self.buffer[0][1]
        s_, done = self.buffer[-1][3], self.buffer[-1][4]
        
        # 移除最旧的转换
        self.buffer.popleft()
        
        return (s, a, R, s_, done)


def read_json(param_path):
    """读取JSON配置文件并返回其内容"""
    try:
        # 确保路径是Path对象
        if isinstance(param_path, str):
            param_path = Path(param_path)

        # 检查文件是否存在
        if not param_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {param_path}")

        # 读取并解析JSON文件
        with open(param_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return config

    except json.JSONDecodeError:
        print(f"错误: 文件 {param_path} 不是有效的JSON格式")
        return None
    except Exception as e:
        print(f"读取配置文件时出错: {e}")
        return None