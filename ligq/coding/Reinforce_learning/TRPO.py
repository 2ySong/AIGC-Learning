#步长的修改本质上是对参数的修改，但是在强化学习里，小参数的更改就会让结果很奇怪，这里我们引入一个变化域，描述更新的策略和旧策略之间的差异，用KL散度，不要让差异太大，从而控制修改的强度

import gymnasium as gym
import numpy as np
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
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, action_dim)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

class ValueNet(nn.Module):
	"""Critic (评论家): 输入状态 -> 输出状态价值 V(s)"""
	def __init__(self, state_dim):
		super(ValueNet, self).__init__()
		self.fc1 = nn.Linear(state_dim, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

# ==========================================
#  第二部分：TRPO 算法主体
# ==========================================

class TRPOAgent:
	def __init__(self, state_dim, action_dim, gamma=0.99, lam=0.95, max_kl=0.01):
		self.device = torch.device("cpu")
		self.gamma = gamma
		self.lam = lam
		self.max_kl = max_kl
		self.damping = 0.1
		self.cg_iters = 10
		self.backtrack_iters = 10
		self.backtrack_coeff = 0.8

		self.actor = PolicyNet(state_dim, action_dim).to(self.device)
		self.critic = ValueNet(state_dim).to(self.device)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

		self.data = []

	def select_action(self, state):
		state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
		logits = self.actor(state)
		dist = Categorical(logits=logits)
		action = dist.sample()
		return action.item()

	def put_data(self, transition):
		self.data.append(transition)

	def compute_gae(self, states, rewards, next_states, dones):
		with torch.no_grad():
			values = self.critic(states).squeeze(1)
			next_values = self.critic(next_states).squeeze(1)

		deltas = rewards + self.gamma * next_values * (1 - dones) - values
		advantages = torch.zeros_like(rewards)

		gae = 0.0
		for t in reversed(range(len(rewards))):
			gae = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * gae
			advantages[t] = gae

		returns = advantages + values
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
		return advantages, returns

	def get_policy_loss(self, states, actions, advantages, old_logits):
		logits = self.actor(states)
		dist = Categorical(logits=logits)
		log_probs = dist.log_prob(actions)
		old_dist = Categorical(logits=old_logits)
		old_log_probs = old_dist.log_prob(actions)
		ratio = torch.exp(log_probs - old_log_probs)
		return (ratio * advantages).mean()

	def get_kl(self, states, old_logits):
		new_logits = self.actor(states)
		new_dist = Categorical(logits=new_logits)
		old_dist = Categorical(logits=old_logits)
		return torch.distributions.kl_divergence(old_dist, new_dist).mean()

	def flat_params(self, model):
		return torch.cat([p.data.view(-1) for p in model.parameters()])

	def set_flat_params(self, model, flat_params):
		idx = 0
		for p in model.parameters():
			length = p.numel()
			p.data.copy_(flat_params[idx:idx + length].view(p.shape))
			idx += length

	def flat_grad(self, grads):
		return torch.cat([g.contiguous().view(-1) for g in grads])

	def fisher_vector_product(self, states, old_logits, vector):
		kl = self.get_kl(states, old_logits)
		grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
		flat_grads = self.flat_grad(grads)
		grad_vector_product = (flat_grads * vector).sum()
		hv = torch.autograd.grad(grad_vector_product, self.actor.parameters())
		flat_hv = self.flat_grad(hv).detach()
		return flat_hv + self.damping * vector

	def conjugate_gradient(self, states, old_logits, b):
		x = torch.zeros_like(b)
		r = b.clone()
		p = r.clone()
		r_dot_r = torch.dot(r, r)

		for _ in range(self.cg_iters):
			Hp = self.fisher_vector_product(states, old_logits, p)
			alpha = r_dot_r / (torch.dot(p, Hp) + 1e-8)
			x = x + alpha * p
			r = r - alpha * Hp
			r_dot_r_new = torch.dot(r, r)
			if r_dot_r_new < 1e-10:
				break
			beta = r_dot_r_new / (r_dot_r + 1e-8)
			p = r + beta * p
			r_dot_r = r_dot_r_new

		return x

	def line_search(self, states, actions, advantages, old_logits, step_dir, old_params, old_loss):
		sHs = torch.dot(step_dir, self.fisher_vector_product(states, old_logits, step_dir))
		if sHs <= 0:
			self.set_flat_params(self.actor, old_params)
			return False, old_params

		max_step = torch.sqrt(2 * self.max_kl / (sHs + 1e-8))
		full_step = max_step * step_dir

		for i in range(self.backtrack_iters):
			step_frac = self.backtrack_coeff ** i
			new_params = old_params + step_frac * full_step
			self.set_flat_params(self.actor, new_params)

			new_loss = self.get_policy_loss(states, actions, advantages, old_logits)
			kl = self.get_kl(states, old_logits)

			if new_loss > old_loss and kl < self.max_kl:
				return True, new_params

		self.set_flat_params(self.actor, old_params)
		return False, old_params

	def update(self):
		if not self.data:
			return

		s, a, r, s_prime, done = zip(*self.data)

		states = torch.tensor(s, dtype=torch.float).to(self.device)
		actions = torch.tensor(a, dtype=torch.long).to(self.device)
		rewards = torch.tensor(r, dtype=torch.float).to(self.device)
		next_states = torch.tensor(s_prime, dtype=torch.float).to(self.device)
		dones = torch.tensor(done, dtype=torch.float).to(self.device)

		advantages, returns = self.compute_gae(states, rewards, next_states, dones)

		for _ in range(80):
			values = self.critic(states).squeeze(1)
			value_loss = F.mse_loss(values, returns)
			self.critic_optimizer.zero_grad()
			value_loss.backward()
			self.critic_optimizer.step()

		with torch.no_grad():
			old_logits = self.actor(states).detach()

		old_loss = self.get_policy_loss(states, actions, advantages, old_logits)
		grads = torch.autograd.grad(old_loss, self.actor.parameters())
		flat_grads = self.flat_grad(grads).detach()

		step_dir = self.conjugate_gradient(states, old_logits, flat_grads)
		old_params = self.flat_params(self.actor).detach()
		self.line_search(states, actions, advantages, old_logits, step_dir, old_params, old_loss)

		self.data = []

# ==========================================
#  第三部分：主循环
# ==========================================

env = gym.make('CartPole-v1')
min_batch_size = 2048
agent = TRPOAgent(state_dim=4, action_dim=2)

print("TRPO 开始训练...")
total_steps = 0
last_final_reward = 0

while total_steps < 100000:
	state, _ = env.reset()
	done = False
	episode_reward = 0

	while not done:
		action = agent.select_action(state)
		next_state, reward, terminated, truncated, _ = env.step(action)
		done = terminated or truncated

		agent.put_data((state, action, reward, next_state, float(done)))

		state = next_state
		episode_reward += reward
		total_steps += 1

		if len(agent.data) >= min_batch_size:
			agent.update()
			print(f"Steps: {total_steps}, Last Complete Reward: {last_final_reward}")

	last_final_reward = episode_reward

print("训练完成！")

