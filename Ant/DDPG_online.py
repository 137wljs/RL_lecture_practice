import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections
import random

algorithm_name = "DDPG"

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

# def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
#     return_list = []
#     for i in range(10):
#         with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
#             for i_episode in range(int(num_episodes/10)):
#                 episode_return = 0
#                 state, _ = env.reset()
#                 done, truncated = False, False
#                 while not done and not truncated:   
#                     action = agent.take_action(state)
#                     action = F.tanh(action.reshape(-1))
#                     next_state, reward, done, truncated, _ = env.step(action.cpu().detach().numpy())
#                     replay_buffer.add(state, action, reward, next_state, done or truncated)
#                     state = next_state
#                     episode_return += reward
#                     if replay_buffer.size() > minimal_size:
#                         b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
#                         transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
#                         agent.update(transition_dict)
#                 return_list.append(episode_return)
#                 if (i_episode+1) % 10 == 0:
#                     pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
#                 pbar.update(1)
#     return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    action = F.tanh(action.reshape(-1))
                    # distinguish whether action is nan
                    if np.isnan(action.cpu().detach().numpy()).any():
                        print("nan action detected")
                    next_state, reward, done, truncated , _ = env.step(action.cpu().detach().numpy())
                    done = truncated or done
                    replay_buffer.add(state, action.detach(), reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1_mu = nn.Linear(state_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        mu = F.relu(self.fc1_mu(x))
        return self.fc2_mu(mu)


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, sigma, tau, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
        self.action_dim = action_dim
        self.device = device

    def take_action(self, state): # 使用固定高斯噪声(sigma为超参数)
        state = torch.FloatTensor([state]).to(self.device)
        action = self.actor(state)
        action = action + self.sigma * torch.randn(self.action_dim).to(self.device)
        return action
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data = param_target.data * (1.0 - self.tau) + param.data * self.tau

    def update(self, transition_dist):
        states = torch.FloatTensor(transition_dist['states']).to(self.device)
        actions = torch.stack(transition_dist['actions']).to(self.device)
        actions = torch.atanh(actions)
        rewards = torch.FloatTensor(transition_dist['rewards']).reshape((-1, 1)).to(self.device)
        next_states = torch.FloatTensor(transition_dist['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dist['dones']).reshape((-1, 1)).to(self.device)
        
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

def train():
    actor_lr = 1e-6
    critic_lr = 1e-3
    buffer_size = 10000
    minimal_size = 1000
    num_episodes = 500
    hidden_dim = 128
    batch_size = 32
    gamma = 0.98
    sigma = 0.2
    tau = 0.005
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_name = "Ant-v4"
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env, max_episode_steps = 200) # 限制最大轮数
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(buffer_size)
    print(f'State dim: {state_dim}, Action dim: {action_dim}')
    agent = DDPG(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, sigma, tau, device)

    return_list = train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)
    episodes_list = list(range(len(return_list)))
    with open(f"{algorithm_name}.txt", "w") as file:
        file.write(str(return_list))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{algorithm_name} on {env_name}')
    plt.savefig(f'{algorithm_name}_training_results.png')

if __name__ == '__main__':
    train()
