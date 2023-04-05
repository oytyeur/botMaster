import torch
import torch.nn as nn
import torch.optim as optim
from random import uniform, random
# from safe_task_execution import Process


# TODO: надо сформировать состояние
# 1. bot_x, bot_y, bot_dir
# 2. goal_x, goal_y
# 3. bot_lin_vel, bot_ang_vel
# 4. frame: включает в себя расстояния до каждой точки на кадре. Принимаем число значений постоянным и равным 200-300
#    так, первое значение является самым левым, последнее - правым
# Возможно добавление маркера о принадлежности незнакомым препятствиям поможет - это ещё 200-300 входов соответственно
# Сначала попробую без маркеров - на сырые лиданые данные

torch.autograd.set_detect_anomaly(True)

# определяем нейронную сеть для политики
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.base = nn.Sequential(*[
        #     nn.Linear(state_dim, 100),
        #     nn.ReLU()])
        # self.mu = nn.Sequential(*[
        #     nn.Linear(100, action_dim),
        #     nn.Tanh()])
        # self.var = nn.Sequential(*[
        #     nn.Linear(100, action_dim),
        #     nn.Softplus()])
        # self.value = nn.Linear(100, 1)
        # self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.fc1 = nn.Linear(state_dim, 75)
        self.fc2 = nn.Linear(75, 50)
        self.mean_layer = nn.Linear(50, action_dim)
        self.std_layer = nn.Linear(50, action_dim)
        self.value_layer1 = nn.Linear(50, 25)
        self.value_layer2 = nn.Linear(25, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward(self, x):
        # base = torch.Tensor(self.base(x)).requires_grad_()
        # print(base.is_leaf)
        # mu = self.mu(base)
        # mu.requires_grad_()
        # print(mu.is_leaf)
        # var = self.var(base).requires_grad_()
        # value = self.value(base).requires_grad_()
        # print(base.grad.data)
        # return mu, var, value

        x = torch.relu(self.fc1(x)).detach()
        x = torch.tanh(self.fc2(x))
        mean = self.mean_layer(x).detach()
        log_std = self.std_layer(x)
        std = torch.exp(log_std).detach()
        # print(std)
        value = torch.relu(self.value_layer1(x))
        value = self.value_layer2(value)
        # print(value)
        return mean, std, value


# определяем функцию обучения с подкреплением методом PPO
def train(env, policy, gamma=0.1, eps_clip=0.5, K=10):
    state = env.reset()
    done = False
    rewards = []
    log_probs = []
    values = []
    entropies = []
    reward = 0
    iterations = 300
    while not done and iterations > 0:
        state_tensor = torch.from_numpy(state).float()
        mean, std, value = policy(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        velocities = action.numpy()

        if velocities[0] > 4.0:
            velocities[0] = 4.0
        elif velocities[0] < 0.1:
            velocities[0] = 0.1

        if velocities[1] > 180.0:
            velocities[1] = 180.0
        elif velocities[1] < -180.0:
            velocities[1] = -180.0

        state, reward, done = env.step(*velocities)

        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value)

        iterations -= 1

    R = 0
    G = []

    for reward in reversed(rewards):
        R = reward + gamma * R
        G.insert(0, R)

    G = torch.tensor(G)
    G = (G - G.mean()) / (G.std() + 1e-5)

    for k in range(K):
        for i in range(len(rewards)):
            state_tensor = torch.from_numpy(state).float()
            mean, std, value = policy(state_tensor)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            ratio = torch.exp(log_prob - log_probs[i])
            surr1 = ratio * G[i]
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * G[i]
            loss = -torch.min(surr1, surr2) + 0.5 * (value - G[i]) ** 2 - 0.01 * dist.entropy()

            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()

    return sum(rewards)


# # создаем среду для игры Pong
# env = gym.make('Pong-v0')

# # создаем экземпляр нейронной сети для политики и обучаем агента
# process = Process()
# policy = Policy()
#
# for i in range(100):
#     episode_reward = train(process, policy)
#     print(f'Episode {i}: Reward {episode_reward}')