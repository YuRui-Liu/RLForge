import sys
import torch
import torch.nn as nn
import numpy as np
import math
import os
from usvlib4ros.dqn.algor.utils import PrioritizedReplayBuffer, FusedNet, NStepBuffer


class DQN:
    """集成了D3QN, PER, CNN, NoisyNets, Checkpointing和TensorBoard的DQN智能体。"""

    def __init__(self, model_path=None,model_path_type=None,
                 use_cnn=True,use_lstm=False, use_double_dqn=True, use_dueling=True,
                 use_per=True, use_noisy_net=True, use_C51=False, use_MSL=False,
                 dim_states=183, dim_actions=5,
                 tau=0.005, lr=1e-4, gama=0.99, batch_size=128, memory_capacity=1e4,
                 epsilon_start=1.0,epsilon_end=0.1,epsilon_decay=500,
                 per_alpha=0.6,per_beta_start=0.4, per_beta_end=1.0,per_beta_deacy_steps=1e5,
                 n_step=3, v_min=-10, v_max=10, atoms=51,lstm_hidden_size=128, lstm_layers=1,
                 device='cpu'
                 ):


        print("--- DQN Agent Configuration ---")
        self.use_cnn = use_cnn
        print(f"Using CNN: {self.use_cnn}")
        self.use_lstm = use_lstm
        print(f"Using LSTM: {self.use_lstm}")
        self.use_double_dqn = use_double_dqn
        print(f"Using Double DQN: {self.use_double_dqn}")
        self.use_dueling = use_dueling
        print(f"Using Dueling Architecture: {self.use_dueling}")
        self.use_per = use_per
        print(f"Using Prioritized Replay (PER): {self.use_per}")
        self.use_noisy_net = use_noisy_net
        print(f"Using Noisy Nets: {self.use_noisy_net}")
        self.use_C51 = use_C51
        print(f"Using Categorical DQN (C51): {self.use_C51}")
        self.use_MSL = use_MSL
        print(f"Using Multi-step Learning (MSL): {self.use_MSL}")
        print("-----------------------------")

        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        
        self.tau = tau
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.memory_capacity = memory_capacity
        self.gama = gama
        self.batch_size = batch_size
        self.device = device
        # 线性探索率参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        # PER相关参数
        self.beta = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_start = per_beta_start
        self.per_beta_deacy_steps = per_beta_deacy_steps
        # C51相关参数
        self.n_step = n_step
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        if self.use_C51:
            self.support = torch.linspace(v_min, v_max, atoms).to(device)
            self.delta_z = (v_max - v_min) / (atoms - 1)
        else:
            self.support = None
            self.delta_z = None
        # 初始化N-step缓冲区
        if self.use_MSL:
            self.n_step_buffer = NStepBuffer(n_step, gama)

        self.eval_net = FusedNet(dim_states, dim_actions, use_cnn,use_lstm, use_dueling, use_noisy_net, use_C51, lstm_hidden_size,lstm_layers, atoms).to(device)
        self.target_net = FusedNet(dim_states, dim_actions, use_cnn,use_lstm, use_dueling, use_noisy_net, use_C51,lstm_hidden_size,lstm_layers, atoms).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        # 计数器
        self.learn_step_counter = 0
        self.total_steps = 0
        self.start_epoch = 0


        if model_path and os.path.exists(model_path):
            print(f"Loading checkpoint from: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=device)
                self.eval_net.load_state_dict(checkpoint['eval_net_state_dict'])
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if model_path_type == "resume":
                    self.start_epoch = checkpoint.get('epoch', 0) + 1
                    self.total_steps = checkpoint.get('total_steps', 0)
                    if not self.use_noisy_net: self.epsilon = checkpoint.get('epsilon', epsilon_start)
                    if self.use_per: self.beta = checkpoint.get('beta', per_beta_start)
                print(f"Checkpoint loaded. Resuming training from Epoch {self.start_epoch}.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                self.start_epoch = 0
        else:
            print("No valid checkpoint found. Starting from scratch.")

        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.target_net.eval()

        if self.use_per:
            self.memory = PrioritizedReplayBuffer(memory_capacity, per_alpha, device)
            # 根据是否使用C51选择损失函数
            if self.use_C51:
                self.loss_func = None  # C51使用自定义损失计算
            else:
                self.loss_func = nn.SmoothL1Loss(reduction='none')
        else:
            self.memory = []
            if self.use_C51:
                self.loss_func = None
            else:
                self.loss_func = nn.MSELoss()

    def save_model(self, epoch, save_dir):
        """保存模型的检查点。"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        state = {
            'epoch': epoch,
            'total_steps': self.total_steps,
            'eval_net_state_dict': self.eval_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'beta': self.beta,
        }
        torch.save(state, filename)
        print(f"Checkpoint saved to {filename}")
    
    def choose_action(self, state, testing=False):  # Add testing flag
        # 在动作选择前重置隐藏状态（可选）
        if self.use_lstm:
            self.eval_net.reset_hidden_states()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Always use eval mode for deterministic action selection
        self.eval_net.eval()

        if self.use_noisy_net and not testing:
            # During training with Noisy Nets, we need noise for exploration.
            # But NoisyLinear's forward pass handles this with self.training.
            # So we set train() mode here for exploration.
            self.eval_net.train()

        if not self.use_noisy_net and not testing and np.random.uniform() < self.epsilon:
            # Epsilon-greedy exploration during training
            action = np.random.randint(0, self.dim_actions)
        else:
            # Deterministic action (exploitation)
            with torch.no_grad():
                if self.use_C51:
                    # C51动作选择：取期望值最大的动作
                    dist = self.eval_net(state)  # [1, A, N]
                    q_values = (dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [1, A]
                    action = q_values.argmax().item()
                else:
                    # 传统DQN动作选择
                    action_values = self.eval_net(state)
                    action = action_values.argmax().item()

        return action
    def store_transition(self, s, a, r, s_, done):

        # 在episode结束时重置隐藏状态
        if done and self.use_lstm:
            self.eval_net.reset_hidden_states()
            self.target_net.reset_hidden_states()
        # 使用N-step缓冲区处理转换
        if self.use_MSL:
            transition = (s, a, r, s_, done)
            n_step_transition = self.n_step_buffer.push(transition)
            if n_step_transition:
                if self.use_per:
                    self.memory.store(n_step_transition)
                else:
                    if len(self.memory) >= self.memory_capacity: self.memory.pop(0)
                    self.memory.append(n_step_transition)
        else:
            # 传统单步存储
            if self.use_per:
                self.memory.store((s, a, r, s_, done))
            else:
                if len(self.memory) >= self.memory_capacity: self.memory.pop(0)
                self.memory.append((s, a, r, s_, done))


    def learn(self):
        # if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if self.use_per:
            self.beta = min(self.per_beta_end,
                             self.per_beta_start  + self.total_steps * (self.per_beta_end -  self.per_beta_start) / self.per_beta_deacy_steps)
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
        else:
            sample_indices = np.random.choice(len(self.memory), self.batch_size)
            mini_batch = [self.memory[i] for i in sample_indices]

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        b_s = torch.FloatTensor(np.array(states)).to(self.device)
        b_a = torch.LongTensor(actions).view(-1, 1).to(self.device)
        b_r = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        b_s_ = torch.FloatTensor(np.array(next_states)).to(self.device)
        b_d = torch.FloatTensor(dones).view(-1, 1).to(self.device)

        if self.use_C51:
            # C51学习算法
            with torch.no_grad():
                next_dist = self.target_net(b_s_)  # [B, A, N]
                
                if self.use_double_dqn:
                    # Double DQN: 使用eval_net选择动作
                    eval_next_q = self.eval_net(b_s_)  # [B, A, N]
                    # 对分布求期望得到Q值
                    eval_q_values = (eval_next_q * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [B, A]
                    next_actions = eval_q_values.argmax(dim=1)  # [B]
                else:
                    # 标准DQN: 使用target_net选择动作
                    target_q_values = (next_dist * self.support.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [B, A]
                    next_actions = target_q_values.argmax(dim=1)  # [B]

                # 正确处理维度扩展
                # next_actions: [B] -> [B, 1, 1] 用于gather操作
                next_actions_expanded = next_actions.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                next_actions_expanded = next_actions_expanded.expand(-1, -1, self.atoms)  # [B, 1, N]
                
                # 提取选中动作的分布
                next_dist = next_dist.gather(1, next_actions_expanded).squeeze(1)  # [B, N]

                # 计算目标分布
                # 使用n-step折扣因子
                gamma_n = self.gama ** (self.n_step if self.use_MSL else 1)
                Tz = b_r + (1 - b_d.float()) * gamma_n * self.support.unsqueeze(0)  # [B, N]
                Tz = Tz.clamp(min=self.v_min, max=self.v_max)
                b_z = (Tz - self.v_min) / self.delta_z  # [B, N]
                
                # 投影计算
                l = b_z.floor().long()  # [B, N]
                u = b_z.ceil().long()   # [B, N]

                # 防止越界
                l = l.clamp(0, self.atoms - 1)
                u = u.clamp(0, self.atoms - 1)

                # 投影到支持点上
                m = torch.zeros(b_s_.size(0), self.atoms).to(self.device)  # [B, N]
                
                # 使用index_add进行投影
                offset = torch.linspace(0, (b_s_.size(0) - 1) * self.atoms, b_s_.size(0)).long().unsqueeze(1).to(self.device)  # [B, 1]
                l_offset = (l + offset).view(-1)  # [B*N]
                u_offset = (u + offset).view(-1)  # [B*N]
                next_dist_flat = next_dist.view(-1)  # [B*N]
                b_z_flat = b_z.view(-1)  # [B*N]
                
                # 投影权重
                l_weight = (u.float() - b_z).view(-1)
                u_weight = (b_z - l.float()).view(-1)
                
                m.view(-1).index_add_(0, l_offset, next_dist_flat * l_weight)
                m.view(-1).index_add_(0, u_offset, next_dist_flat * u_weight)

            # 获取当前状态的动作分布
            dist = self.eval_net(b_s)  # [B, A, N]
            actions_expanded = b_a.unsqueeze(-1).expand(-1, -1, self.atoms)  # [B, 1, N]
            dist = dist.gather(1, actions_expanded).squeeze(1)  # [B, N]
            
            # 计算交叉熵损失
            loss = -(m * torch.log(dist + 1e-8)).sum(dim=1)  # [B]

            if self.use_per:
                td_errors = loss.detach()
                weighted_loss = (loss * is_weights).mean()
                self.memory.update_priorities(idxs, td_errors)
                final_loss = weighted_loss
            else:
                final_loss = loss.mean()

        else:
            # 传统DQN学习算法
            q_eval = self.eval_net(b_s).gather(1, b_a)
            q_next = self.target_net(b_s_).detach()
            if self.use_double_dqn:
                best_actions = self.eval_net(b_s_).argmax(dim=1).view(-1, 1)
                q_target_next = q_next.gather(1, best_actions)
            else:
                q_target_next = q_next.max(1)[0].view(-1, 1)
            
            # 使用n-step折扣因子
            gamma_n = self.gama ** (self.n_step if self.use_MSL else 1)
            q_target = b_r + (1 - b_d.float()) * gamma_n * q_target_next

            if self.use_per:
                loss = self.loss_func(q_eval, q_target)
                td_errors = loss.squeeze()
                weighted_loss = (loss * is_weights.unsqueeze(1)).mean()
                self.memory.update_priorities(idxs, td_errors)
                final_loss = weighted_loss
            else:
                loss = self.loss_func(q_eval, q_target)
                final_loss = loss

        self.optimizer.zero_grad()
        final_loss.backward()
        # 【可选】添加梯度裁剪防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 10)
        self.optimizer.step()

        # 对目标网络进行软更新
        with torch.no_grad():
            for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(self.tau * eval_param.data + (1.0 - self.tau) * target_param.data)

        if self.use_noisy_net:
            self.eval_net.reset_noise()
            self.target_net.reset_noise()

        # 返回损失值用于记录
        return final_loss.item()

    def decay_epsilon(self):
        self.total_steps += 1
        self.epsilon = self.epsilon_end  + (self.epsilon_start - self.epsilon_end ) * math.exp(-1. * self.total_steps / self.epsilon_decay)
