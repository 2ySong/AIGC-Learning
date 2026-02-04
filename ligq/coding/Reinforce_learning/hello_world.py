import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import torch.optim as optim

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    
    def push(self,state,action,reward,next_state,done):
        return self.buffer.append((state,action,reward,next_state,done))
    
    def sample(self,batch_size):
        batch=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*batch)
        return np.array(state),action,reward,np.array(next_state),done
    
    def __len__(self):
        return len(self.buffer)
    
class QNetwork(nn.Module):
    def __init__(self, state_dim,action_dim):
        super().__init__()
        #推车游戏有四个状态，动作有两个
    
        self.fc1=nn.Linear(state_dim,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,action_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))

        return self.fc3(x)
    
class DQNAgent:
    def __init__(self,state_dim,action_dim,learning_rate=1e-3):
        self.action_dim=action_dim
        self.state_dim=state_dim
        self.gamma=0.99
        self.memory=ReplayBuffer(10000)
        self.device = torch.device("cpu")

        self.policy_net=QNetwork(state_dim,action_dim).to(self.device)
        self.target_net=QNetwork(state_dim,action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())#刚开始要参数同步

        self.target_net.eval()

        self.optimizer=optim.Adam(self.policy_net.parameters(),lr=learning_rate)
        self.epsilon=1
        self.epsilon_decay=0.995
        self.epsilon_min=0.01

    def select_action(self,state):
        if random.random()<self.epsilon:
            return random.randrange(self.action_dim)
        
        else:
            with torch.no_grad():
                state_t=torch.FloatTensor(state).unsqueeze(0).to(self.device)

                q_values=self.policy_net(state_t)

                return q_values.argmax().item()
        
    def update_epsilon(self):
        self.epsilon=max(self.epsilon_min,self.epsilon*self.epsilon_decay)

    def update(self,batch_size=64):
        if len(self.memory)<batch_size:
            return
        
        state,action,reward,next_state,done=self.memory.sample(batch_size)
        state_batch=torch.FloatTensor(state).to(self.device)
        next_state_batch=torch.FloatTensor(next_state).to(self.device)
        reward_batch=torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        action_batch=torch.LongTensor(action).unsqueeze(1).to(self.device)
        done_batch=torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_values=self.policy_net(state_batch)
        curr_q_value=q_values.gather(1,action_batch)

        with torch.no_grad():
            next_q_values=self.target_net(next_state_batch)
            max_next_q_value=next_q_values.max(1)[0].unsqueeze(1)
            expected_q_value=reward_batch+self.gamma*max_next_q_value*(1-done_batch)
        
        loss=torch.nn.functional.mse_loss(curr_q_value,expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# 初始化环境和智能体
env = gym.make('CartPole-v1')
agent = DQNAgent(state_dim=4, action_dim=2)

print("开始训练...")
rewards = []

for episode in range(500): # 玩 500 局
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(1000):
        # 1. 选动作 (互动)
        action = agent.select_action(state)
        
        # 2. 执行动作 (环境反馈)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 3. 存记忆 (积累经验)
        agent.memory.push(state, action, reward, next_state, done)
        
        # 4. 学习 (变聪明!)
        # 注意：这里是每一帧都学一次。有些实现是每隔几帧学一次。
        agent.update(batch_size=64)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
            
    # 5. 更新 Epsilon (慢慢减少乱走)
    agent.update_epsilon()
    rewards.append(total_reward)
    
    # 6. 每隔 20 局，同步一下 Target Net
    if episode % 20 == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    

print("训练结束！")
