import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# ==========================================
#  第一部分：定义神经网络 (Actor 和 Critic)
# ==========================================

class PolicyNet(nn.Module):
    """Actor (演员): 输入状态 -> 输出动作概率"""
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # ★ 关键点 1: 输出是 Softmax 概率分布，而不是 Q 值
        x = F.softmax(self.fc2(x), dim=1)
        return x

class ValueNet(nn.Module):
    """Critic (评论家): 输入状态 -> 输出状态价值 V(s)"""
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1) # 输出只是一个标量

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ==========================================
#  第二部分：PPO 算法主体
# ==========================================

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, clip_param=0.2):
        self.device = torch.device("cpu") # CartPole 用 CPU 够快了
        self.gamma = gamma
        self.clip_param = clip_param # ★ PPO 的截断参数 (0.2)
        
        # 初始化两个网络
        self.actor = PolicyNet(state_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim).to(self.device)
        
        # PPO 通常用 Adam
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # ★ 关键点 2: Rollout Buffer (一次性内存)
        # 我们需要存：状态、动作、奖励、以及"旧的动作概率"
        self.data = [] 

    def select_action(self, state):
        """与环境交互时使用"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 1. 预测概率
        probs = self.actor(state)
        
        # 2. 创建分布并采样
        dist = Categorical(probs)
        action = dist.sample() # 按照概率抽样，而不是 argmax
        
        # 3. 计算 log_prob (为了后续计算比率)
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def put_data(self, transition):
        """存一条数据 (s, a, r, s', done, log_prob)"""
        self.data.append(transition)

    def update(self):
        """PPO 的核心更新逻辑"""
        # 1. 把 buffer 里的东西拿出来整理
        s, a, r, s_prime, done_mask, old_log_prob = zip(*self.data)
        
        # 转成 Tensor
        s = torch.tensor(s, dtype=torch.float).to(self.device)
        a = torch.tensor(a, dtype=torch.long).to(self.device).view(-1, 1)
        r = torch.tensor(r, dtype=torch.float).to(self.device).view(-1, 1)
        old_log_prob = torch.tensor(old_log_prob, dtype=torch.float).to(self.device).view(-1, 1)
        
        # 2. 计算"真实回报" (Monte Carlo Returns)
        # 这里用简化的倒推法计算 Target Value
        returns = []
        G = 0
        for reward in reversed(r):
            G = reward + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float).to(self.device).view(-1, 1)
        
        # 归一化 returns 可以让训练更稳 (Trick)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # 3. PPO 反复刷数据 (K epochs)
        # DQN 只能更一次，PPO 这里可以循环很多次
        for _ in range(10): 
            # --- 重新计算当前的 V 和 Pi ---
            # 这里的 actor 是最新的，old_log_prob 是旧的
            probs = self.actor(s)
            dist = Categorical(probs)
            log_prob = dist.log_prob(a.squeeze()).view(-1, 1) # 新的 log_prob
            entropy = dist.entropy().mean() # 鼓励探索
            
            values = self.critic(s)
            
            # --- 计算优势函数 (Advantage) ---
            # Advantage = 实际回报 - Critic预测的值
            # detach() 很重要：我们只更新 Actor，不希望梯度传给 Critic
            advantage = returns - values.detach()

            # --- ★ PPO 核心公式开始 ★ ---
            
            # Ratio = exp(log_new - log_old) = pi_new / pi_old
            ratio = torch.exp(log_prob - old_log_prob)
            
            # 第一项：原始冲动
            surr1 = ratio * advantage
            
            # 第二项：截断保护 (Clip)
            # 如果 ratio 超过 1+clip (1.2) 或 低于 1-clip (0.8)，就强行截断
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            
            # Actor Loss = - min(surr1, surr2)
            # 我们要最大化目标，所以 Loss 取负号
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            
            # Critic Loss = MSE(预测值, 真实回报)
            critic_loss = F.mse_loss(values, returns)
            
            # --- 梯度更新 ---
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # ★ 关键点 3: 学完必须清空 Buffer！不能留到下一局
        self.data = [] 

# ==========================================
#  第三部分：主循环
# ==========================================

env = gym.make('CartPole-v1')
agent = PPOAgent(state_dim=4, action_dim=2)

print("PPO 开始训练...")
for episode in range(600): # 玩 600 局
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # 1. 选动作 (同时拿到 log_prob)
        action, log_prob = agent.select_action(state)
        
        # 2. 执行
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 3. 存数据 (注意存入了 log_prob)
        # PPO 是 On-Policy，所以我们不需要存 next_state 用于更新(除非用 GAE)
        # 但为了格式统一，这里可以随便存个 next_state
        trans = (state, action, reward, next_state, done, log_prob)
        agent.put_data(trans)
        
        state = next_state
        total_reward += reward
    
    # 4. ★ 一局结束，立马学习！
    agent.update()
    
    if episode % 20 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}")
        if total_reward > 200:
            print("起飞！🚀")

print("训练完成！")