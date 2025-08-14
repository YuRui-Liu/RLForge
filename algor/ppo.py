import sys
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import torch.optim as optim
import torch.nn.functional as F

# PPO 策略网络 (Actor)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        # log_std 用于输出动作分布的标准差。通常是一个可学习的参数，与状态无关或与状态有关。
        # 这里使用一个简单的，与状态无关的可学习参数。
        # 初始化为0，exp(0)=1，即初始标准差为1。
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x))  # 常用tanh将均值限制在-1到1之间，方便后续动作缩放

        # 确保标准差为正值。通过exp(log_std)得到。
        std = torch.exp(self.log_std)

        # 返回一个正态分布对象，用于采样和计算log_prob
        return Normal(mean, std.expand_as(mean))  # std需要扩展到与mean相同的形状


# PPO 价值网络 (Critic)
class VCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(VCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)  # 输出一个标量价值

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc_out(x)


# ====================================================================================================
# PPOAgent 类 (完善用户提供的代码)
# ====================================================================================================

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4,
                 gamma=0.99, gae_lambda=0.95, clip_param=0.2, ppo_epochs=10, ent_coef=0.01,device="cpu"):

        self.gamma, self.gae_lambda, self.clip_param = gamma, gae_lambda, clip_param
        self.ppo_epochs, self.ent_coef = ppo_epochs, ent_coef
        self.device = device
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = VCritic(state_dim, hidden_dim).to(device)
        # 优化器同时优化Actor和Critic的参数
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def select_action(self, state):
        # 确保状态是浮点型张量且有批次维度 (1, state_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist = self.actor(state_t)
        action = dist.sample()  # 从分布中采样动作
        log_prob = dist.log_prob(action).sum(-1)  # 计算动作的对数概率，多维动作需要求和
        # 返回分离的动作和对数概率，并去除批次维度 (action_dim,) 和标量
        return action.detach().squeeze(0), log_prob.detach().squeeze(0)

    def update(self, agent_data):
        # agent_data 包含一次rollout收集到的数据：
        # (states, actions, old_log_probs, rewards, dones, last_state_value)
        # states: rollout中每个时间步的状态 (N_steps, state_dim)
        # actions: rollout中每个时间步采取的动作 (N_steps, action_dim)
        # old_log_probs: rollout中每个时间步动作的旧策略对数概率 (N_steps,)
        # rewards: rollout中每个时间步获得的奖励 (N_steps,)
        # dones: rollout中每个时间步是否终止的标志 (N_steps,)
        # last_state_value: rollout结束后，最后一个状态的价值 (如果回合未结束，则为V(s_final)，否则为0) (标量)
        states, actions, old_log_probs, rewards, dones, last_state_value = agent_data

        # 将Numpy数组转换为PyTorch张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        last_state_value = torch.FloatTensor([last_state_value]).to(self.device)  # 确保是张量

        actor_losses = []
        critic_losses = []
        total_losses = []
        entropy_losses = []

        # 计算当前rollout中所有状态的价值 (V(s_0), ..., V(s_{N_steps-1}))
        with torch.no_grad():
            values = self.critic(states).squeeze(-1)  # (N_steps,)
            # 将最后一个状态的价值添加到values的末尾，用于GAE计算中的V(s_{t+1})
            # 这样 values_all 就包含了 V(s_0), ..., V(s_{N_steps-1}), V(s_N)
            values_all = torch.cat((values, last_state_value))  # (N_steps + 1,)

        # 计算优势 (Advantages) 和回报 (Returns)
        # advantages 和 returns 都对应 rollout 中的 N_steps 个状态
        advantages, returns = self._calculate_gae(rewards, values_all, dones)

        # 进行PPO_epochs次策略更新
        for _ in range(self.ppo_epochs):
            # 再次计算当前状态的价值预测，用于Critic Loss（因为Critic网络参数已更新）
            current_values_pred = self.critic(states).squeeze(-1)  # (N_steps,)

            # 获取当前策略下的动作分布和对数概率
            dist = self.actor(states)
            entropy = dist.entropy().mean()  # 熵损失，鼓励探索
            new_log_probs = dist.log_prob(actions).sum(-1)  # (N_steps,)

            # 计算重要性采样比率
            ratio = torch.exp(new_log_probs - old_log_probs.detach())

            # 计算PPO的裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()  # 策略损失，最小化是因为梯度上升，转为最小化负值

            # 价值函数损失
            critic_loss = F.mse_loss(current_values_pred, returns)  # Critic拟合蒙特卡洛回报或GAE回报

            # 总损失 = 策略损失 + 0.5 * 价值损失 - 熵损失系数 * 熵损失
            total_loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy

            # 梯度清零、反向传播、优化器步进
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # 记录损失以便返回平均值
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            total_losses.append(total_loss.item())
            entropy_losses.append(entropy.item())

        # 返回平均损失
        return np.mean(actor_losses), np.mean(critic_losses), np.mean(total_losses), np.mean(entropy_losses)

    def _calculate_gae(self, rewards, values_all, dones):
        # GAE (Generalized Advantage Estimation) 计算
        # rewards: rollout中N_steps个奖励 (N_steps,)
        # values_all: rollout中N_steps个状态的价值 + 最后一个状态的价值 (N_steps + 1,)
        #             V(s_0), ..., V(s_{N_steps-1}), V(s_N)
        # dones: rollout中N_steps个done标志 (N_steps,)

        advantages = torch.zeros_like(rewards).to(self.device)  # 初始化优势为0
        last_gae_lam = 0  # 上一个时间步的GAE值，用于迭代计算

        # 从后往前计算GAE
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]  # 如果当前步是done，则下一状态无价值

            current_value = values_all[t]  # 当前状态的价值 V(s_t)
            next_value = values_all[t + 1]  # 下一个状态的价值 V(s_{t+1}) (包括了rollout的最终状态价值)

            # TD残差 (delta) = reward + gamma * V(s_{t+1}) * (1-done) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - current_value

            # GAE递推公式
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

        # 计算Returns (Q值估计) = 优势 + 价值函数估计
        # 这里返回的returns是用于critic更新的目标值，对应rollout中的N_steps个状态
        returns = advantages + values_all[:-1]  # values_all[:-1] 对应 V(s_0), ..., V(s_{N_steps-1})

        # 对优势进行标准化，有助于训练稳定
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages.detach(), returns.detach()  # 返回分离的张量，避免梯度流回GAE计算