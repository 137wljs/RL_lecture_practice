import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

algorithm_name = "cnn_PPO"

class CNNNet(nn.Module):
    def __init__(self, action_dim):
        super().__init__()
        # 第一卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二卷积层
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三卷积层
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc_mu = nn.Linear(128, action_dim)
        self.fc_sigma = nn.Linear(128, action_dim)
        self.fc_value = nn.Linear(128, 1)

    def forward(self, x):
        # 输入 x 的形状为 (batch_size, 3, 480, 480)

        # 第一卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # 第二卷积块
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # 第三卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # 展平
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # 策略和价值输出
        mu = self.fc_mu(x)
        sigma = F.softplus(torch.clamp(self.fc_sigma(x), -20, 2)) + 1e-6
        value = self.fc_value(x)

        return mu, sigma, value


class PPO:
    def __init__(self, action_dim, lr, lmbda, epochs, eps, gamma, device):
        self.net = CNNNet(action_dim).to(device)
        self.net_optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state): # 改成从高斯分布中取样
        mean, std, _ = self.net(state)
        action_dist = torch.distributions.Normal(mean.view(-1), std.view(-1))
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        log_prob = torch.sum(log_prob) # 这个动作的概率
        return action, log_prob

    def gae(self, td_delta):
        td_delta = td_delta.detach().numpy()
        advantages_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantages_list.append(advantage)
        advantages_list.reverse()
        return torch.FloatTensor(advantages_list)

    def update(self, transition_dist):
        states = torch.stack(transition_dist['states']).to(self.device)
        actions = torch.stack(transition_dist['actions']).to(self.device)
        actions = torch.atanh(actions)
        old_log_probs = torch.stack(transition_dist['log_probs']).to(self.device).detach()
        rewards = torch.FloatTensor(transition_dist['rewards']).reshape((-1, 1)).to(self.device)
        next_states = torch.stack(transition_dist['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dist['dones']).reshape((-1, 1)).to(self.device)
        _, _, value = self.net(states)
        _, _, next_value = self.net(next_states)
        td_target = rewards + self.gamma * next_value * (1 - dones)
        td_delta = td_target - value
        advantage = self.gae(td_delta.cpu()).to(self.device)

        for _ in range(self.epochs):
            means, stds, _ = self.net(states)
            action_dists = torch.distributions.Normal(means, stds)
            log_probs = action_dists.log_prob(actions)
            log_probs = torch.sum(log_probs, dim = 1, keepdim = True)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            _, _, value = self.net(states)
            loss = torch.mean(-torch.min(surr1, surr2)) + torch.mean(F.mse_loss(value, td_target.detach()))
            self.net_optimizer.zero_grad()
            loss.backward()
            self.net_optimizer.step()


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train():
    lr = 2e-6
    num_episodes = 500
    gamma = 0.98
    lmbda = 0.5
    epochs = 10
    eps = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_name = "Ant-v4"
    env = gym.make(env_name, render_mode = 'rgb_array')
    env = gym.wrappers.TimeLimit(env, max_episode_steps = 200) # 限制最大轮数
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f'State dim: {state_dim}, Action dim: {action_dim}')
    agent = PPO(action_dim, lr, lmbda, epochs, eps, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'log_probs': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()
                done, truncated = False, False
                while not done and not truncated:
                    image = env.render().transpose(2, 0, 1)
                    image = torch.FloatTensor(image.copy()).to(device)
                    image /= 255
                    transition_dict['states'].append(image)
                    action, log_prob = agent.take_action(image.view(1, 3, 480, 480))
                    action = F.tanh(action.reshape(-1))
                    next_state, reward, done, truncated, _ = env.step(action.cpu().detach().numpy())
                    next_image = env.render().transpose(2, 0, 1)
                    next_image = torch.FloatTensor(next_image.copy()).to(device)
                    next_image /= 255
                    done = done or truncated
                    transition_dict['actions'].append(action)
                    transition_dict['log_probs'].append(log_prob)
                    transition_dict['next_states'].append(next_image)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

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
