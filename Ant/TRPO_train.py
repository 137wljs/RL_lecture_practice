import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

algorithm_name = "basic_TRPO"

class PolicyNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1_mu = nn.Linear(state_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, action_dim)
        self.fc1_sigma = nn.Linear(state_dim, hidden_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        mu = F.relu(self.fc1_mu(x))
        sigma = F.relu(self.fc1_sigma(x))
        return self.fc2_mu(mu), F.softplus(torch.clamp(self.fc2_sigma(sigma), -20, 2)) + 1e-6


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TRPO:
    def __init__(self, state_dim, hidden_dim, action_dim, critic_lr, lmbda, kl_div_p, alpha, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_div_p = kl_div_p
        self.alpha = alpha
        self.device = device

    def take_action(self, state): # 改成从高斯分布中取样
        state = torch.FloatTensor([state]).to(self.device)
        mean, std = self.actor(state)
        action_dist = torch.distributions.Normal(mean, std)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim = 1) # 这个动作的概率
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
    
    def hessian_matrix_vector_product(self,
                                    states,
                                    old_action_dists,
                                    vector,
                                    damping=0.1):
        mu, std = self.actor(states)
        new_action_dists = torch.distributions.Normal(mu, std)
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat(
            [grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector + damping * vector

    def conjugate_gradient(self, grad, states, old_action_dists):
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15):
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            mu, std = new_actor(states)
            new_action_dists = torch.distributions.Normal(mu, std)
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                     new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_div_p:
                return new_para
        return old_para

    def policy_learn(self, states, actions, old_action_dists, old_log_probs,
                     advantage):
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)
        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_div_p /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantage, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor.parameters())

    def update(self, transition_dist):
        states = torch.FloatTensor(transition_dist['states']).to(self.device)
        actions = torch.stack(transition_dist['actions']).to(self.device)
        actions = torch.atanh(actions)
        old_log_probs = torch.stack(transition_dist['log_probs']).to(self.device).detach()
        rewards = torch.FloatTensor(transition_dist['rewards']).reshape((-1, 1)).to(self.device)
        next_states = torch.FloatTensor(transition_dist['next_states']).to(self.device)
        dones = torch.FloatTensor(transition_dist['dones']).reshape((-1, 1)).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.gae(td_delta.cpu()).to(self.device)
        
        means, stds = self.actor(states)
        old_action_dists = torch.distributions.Normal(means.detach(),
                                                      stds.detach())
        old_log_probs = old_action_dists.log_prob(actions)
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs,
                          advantage)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train():
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.5
    kl_div_p = 0.01
    alpha = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env_name = "Ant-v4"
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)
    env = gym.wrappers.TimeLimit(env, max_episode_steps = 200) # 限制最大轮数
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f'State dim: {state_dim}, Action dim: {action_dim}')
    agent = TRPO(state_dim, hidden_dim, action_dim, critic_lr, lmbda, kl_div_p, alpha, gamma, device)

    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'log_probs': [], 'next_states': [], 'rewards': [], 'dones': []}
                state, _ = env.reset()
                done, truncated = False, False
                while not done and not truncated:
                    # env.render()
                    action, log_prob = agent.take_action(state)
                    action = F.tanh(action.reshape(-1))
                    next_state, reward, done, truncated, _ = env.step(action.cpu().detach().numpy())
                    done = done or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['log_probs'].append(log_prob)
                    transition_dict['next_states'].append(next_state)
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
