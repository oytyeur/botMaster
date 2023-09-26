import torch
import torch.nn as nn
import torch.optim as optim
from random import uniform, random
import time
from math import degrees

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda')


# Класс Policy определяет предыдущую структуру сети,
# где определён порядок двойной нейросети: для полезности и политики,
# однако из конечной архитектуры сети, данный класс исключён и заменён классом PolicyNew
# Конечная структура потребовала часть исходной уже обученной исходной сети,
# в связи чем возникла потребность копирования только требуемой части,
# для чего и созданы вспомогательыне методы работы с параметрами

'''
# # определяем нейронную сеть для политики
class Policy(nn.Module):
    # TODO: добавить новую архитектуру
    def __init__(self, lidar_frames_num):
        super(Policy, self).__init__()
        self.conv1 = torch.nn.Conv1d(lidar_frames_num, 32, (5, ), stride=(2,))
        # self.conv2 = torch.nn.Conv1d(32, 16, (3, ), stride=(2,))
        self.fc_lidar = torch.nn.Linear(1984, 256)
        self.fc_full_state = torch.nn.Linear(260, 128)  ################################
        self.fc_state_to_act = torch.nn.Linear(128, 64)  ################################
        # self.mu_lin_v = torch.nn.Linear(128, 1)
        self.mu_ang_w = torch.nn.Linear(64, 1)  ##########################################
        self.logstd_ang_w = torch.nn.Linear(64, 1)
        # self.mu_lin_v = torch.nn.Linear(64, 1)
        # self.mu_ang_w = torch.nn.Linear(64, 1)
        # self.log_std = torch.nn.Linear(128, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(1))

        self.pi_value_1 = torch.nn.Linear(128, 64) #################################################
        self.pi_value_2 = torch.nn.Linear(64, 64)#############################################
        self.pi_value_3 = torch.nn.Linear(64, 1)  ###################################

        self.policy_optimizer = optim.Adam(self.parameters(), lr=0.0002)
        self.value_optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.actions = []

        self.old_log_probs = []
        self.log_probs = []

        self.rewards = []

        self.target_values = []
        self.pi_values = []
        # self.new_pi_values = []

        self.entropies = []

    def get_params(self):
        return self.fc_full_state.weight, self.fc_full_state.bias, \
               self.fc_state_to_act.weight, self.fc_state_to_act.bias, \
               self.mu_ang_w.weight, self.mu_ang_w.bias, \
               self.pi_value_1.weight, self.pi_value_1.bias, \
               self.pi_value_2.weight, self.pi_value_2.bias, \
               self.pi_value_3.weight, self.pi_value_3.bias

    def forward(self, state):

        st_vels = torch.from_numpy(state[0]).float()
        st_goal = torch.from_numpy(state[1]).float()
        st_lidar = torch.from_numpy(state[2]).float()

        lidar_feat = torch.relu(self.conv1(st_lidar))
        # lidar_feat = torch.relu(self.conv2(lidar_feat))
        lidar_feat = torch.flatten(lidar_feat)
        lidar_feat = torch.relu(self.fc_lidar(lidar_feat))
        zez = torch.zeros(256)
        lidar_feat = zez
        full_state = torch.concat((lidar_feat, st_vels, st_goal))
        full_state_2 = torch.tanh(self.fc_full_state(full_state))

        # mean_lin_v = torch.tanh(self.mu_lin_v(full_state_2))  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        state_to_act = torch.relu(self.fc_state_to_act(full_state_2))
        mean_ang_w = torch.tanh(self.mu_ang_w(state_to_act))
        # mean_lin_v = torch.sigmoid(self.mu_lin_v(state_to_mu))
        # mean_ang_w = torch.tanh(self.mu_ang_w(state_to_mu))

        # act_log_std = torch.tanh(self.logstd_ang_w(state_to_act))
        act_log_std = self.log_std

        # pi_value = torch.relu(self.pi_value(full_state_2))  ########### ДЛЯ ДИСТИЛЛЯЦИИ
        pi_value = torch.relu(self.pi_value_1(full_state_2))
        pi_value = torch.relu(self.pi_value_2(pi_value))
        pi_value = self.pi_value_3(pi_value)

        return 0, mean_ang_w, act_log_std, pi_value
'''


class PolicyNew(nn.Module):
    def __init__(self, lidar_frame_size):
        super(PolicyNew, self).__init__()

        self.fc_lidar = torch.nn.Linear(lidar_frame_size, 256)
        self.fc_full_state = torch.nn.Linear(260, 128)

        self.fc_state_to_act = torch.nn.Linear(128, 64)
        self.mu_ang_w = torch.nn.Linear(64, 1)

        self.pi_value_1 = torch.nn.Linear(128, 64)
        self.pi_value_2 = torch.nn.Linear(64, 64)
        self.pi_value_3 = torch.nn.Linear(64, 1)  ## ЭТО ДЛЯ ДИСТИЛЛЯЦИИ
        #

        self.policy_optimizer = optim.Adam(self.parameters(), lr=0.00001)
        self.value_optimizer = optim.Adam(self.parameters(), lr=0.0001)

        self.actions = []

        self.old_log_probs = []
        self.log_probs = []

        self.rewards = []

        self.target_values = []
        self.pi_values = []
        # self.new_pi_values = []

        self.entropies = []

    def forward(self, state):

        st_vels = torch.from_numpy(state[0]).float()
        st_goal = torch.from_numpy(state[1]).float()

        st_lidar = torch.from_numpy(state[2]).float()
        lidar_feat = torch.relu(self.fc_lidar(st_lidar))

        full_state = torch.concat((lidar_feat, st_vels, st_goal))
        full_state = torch.tanh(self.fc_full_state(full_state))

        state_to_act = torch.relu(self.fc_state_to_act(full_state))
        mean_ang_w = torch.tanh(self.mu_ang_w(state_to_act))

        pi_value = torch.relu(self.pi_value_1(full_state))
        pi_value = torch.relu(self.pi_value_2(pi_value))
        pi_value = self.pi_value_3(pi_value)

        return 0, mean_ang_w, 0, pi_value

    def get_params(self):
        return self.fc_full_state.weight, self.fc_full_state.bias, \
               self.fc_state_to_act.weight, self.fc_state_to_act.bias, \
               self.mu_ang_w.weight, self.mu_ang_w.bias, \
               self.pi_value_1.weight, self.pi_value_1.bias, \
               self.pi_value_2.weight, self.pi_value_2.bias, \
               self.pi_value_3.weight, self.pi_value_3.bias

    def set_param(self, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):
        self.fc_full_state.weight = w1
        self.fc_full_state.bias = b1

        self.fc_state_to_act.weight = w2
        self.fc_state_to_act.bias = b2

        self.mu_ang_w.weight = w3
        self.mu_ang_w.bias = b3

        self.pi_value_1.weight = w4
        self.pi_value_1.bias = b4

        self.pi_value_2.weight = w5
        self.pi_value_2.bias = b5

        self.pi_value_3.weight = w6
        self.pi_value_3.bias = b6


class Value(nn.Module):
    # TODO: НУ ВОТКНЁМ, ПОСМОТРИМ
    def __init__(self, lidar_frames_num):
        super(Value, self).__init__()
        self.conv1 = torch.nn.Conv1d(lidar_frames_num, 32, (5,), stride=(2,))
        self.conv2 = torch.nn.Conv1d(32, 16, (3,), stride=(2,))
        self.fc_lidar = torch.nn.Linear(480, 256)
        self.fc_full_state = torch.nn.Linear(260, 128)

        self.v_value_1 = torch.nn.Linear(128, 64)
        self.v_value_2 = torch.nn.Linear(64, 1)  ## ЭТО ДЛЯ ДИСТИЛЛЯЦИИ
        # self.v_value = torch.nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

        self.v_values = []

        self.target_values = []

    def forward(self, state):
        st_vels = torch.from_numpy(state[0]).float()
        st_goal = torch.from_numpy(state[1]).float()
        st_lidar = torch.from_numpy(state[2]).float()

        lidar_feat = torch.relu(self.conv1(st_lidar))
        lidar_feat = torch.relu(self.conv2(lidar_feat))
        lidar_feat = torch.flatten(lidar_feat)
        lidar_feat = torch.relu(self.fc_lidar(lidar_feat))
        full_state = torch.concat((lidar_feat, st_vels, st_goal))
        full_state_2 = torch.relu(self.fc_full_state(full_state))

        # v_value = torch.relu(self.v_value(full_state_2))  ########### ДЛЯ ДИСТИЛЛЯЦИИ
        v_value = torch.relu(self.v_value_1(full_state_2))
        v_value = self.v_value_2(v_value)

        return v_value


def try_policy(policy_net, state, std=0.1):
    _, ang_w_mean, log_std, pi_value = policy_net(state)
    dist_w = torch.distributions.normal.Normal(ang_w_mean, std)

    entropy_w = dist_w.entropy()
    entropy = entropy_w
    act_w = dist_w.sample()

    w_log_prob = dist_w.log_prob(act_w)
    log_prob = w_log_prob

    return log_prob, pi_value, entropy


def try_value(value_net, state):
    v_value = value_net(state)
    return v_value


def train(process, policy_net, e_p, e_v,
          clip_eps=0.2, ent_coef=0.01, std=0.75):
    # Сбор данных агентом
    with torch.no_grad():
        policy_net.actions, \
        states, \
        policy_net.old_log_probs, \
        rewards, \
        pi_values, \
        v_values, \
        policy_net.entropies, \
        total_reward, \
        empty_set = run_actor(process, policy_net, std=std)

        if empty_set:
            return None

        gaes, norm_gaes = calculate_gaes(rewards, pi_values)

        policy_net.target_values = calculate_discounts(rewards)

    print("REWARD", total_reward)

    # ОПТИМИЗАЦИЯ ЦЕЛЕВОЙ ФУНКЦИИ П-СЕТИ ПОЛИТИКИ
    for _ in range(e_p):
        new_log_probs = torch.zeros(len(states))
        new_pi_values = torch.zeros(len(states))
        new_entropies = torch.zeros(len(states))
        for t in range(len(states)):
            new_log_prob, new_pi_value, new_entropy = try_policy(policy_net, states[t], std=std)
            new_log_probs[t] = new_log_prob
            new_pi_values[t] = new_pi_value
            new_entropies[t] = new_entropy

        policy_net.log_probs = new_log_probs
        policy_net.entropies = new_entropies

        log_ratios = policy_net.log_probs - policy_net.old_log_probs

        ratios = torch.exp(log_ratios)

        cpi_loss = gaes * ratios
        clip_loss = gaes * torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps)
        policy_loss = -torch.min(cpi_loss, clip_loss).mean()
        entropy_loss = policy_net.entropies.mean()

        full_loss = (policy_loss + ent_coef * ent_coef * entropy_loss).mean()
        policy_net.policy_optimizer.zero_grad()
        full_loss.backward()
        policy_net.policy_optimizer.step()

    # ОПТИМИЗАЦИЯ ЦЕЛЕВОЙ ФУНКЦИИ V-СЕТИ ПОЛЕЗНОСТИ
    for _ in range(e_v):
        new_pi_values = torch.zeros(len(states))
        for t in range(len(states)):
            _, new_pi_value, _ = try_policy(policy_net, states[t])
            new_pi_values[t] = new_pi_value

        policy_net.pi_values = new_pi_values
        value_loss = 0.5 * ((policy_net.target_values - policy_net.pi_values) ** 2).mean()
        policy_net.value_optimizer.zero_grad()
        value_loss.backward()
        policy_net.value_optimizer.step()

    return total_reward


# Сбор данных за эпизод
def run_actor(process, policy, value, std=0.1, testing=False):

    state = process.reset()
    st_vel = state[0]

    actions = []
    states = []
    log_probs = []
    rewards = []
    pi_values = []
    v_values = []
    entropies = []

    done = False
    empty_set = False

    iterations = 100
    while not done and iterations > 0:
        _, ang_w_mean, log_std, pi_value = policy(state)
        dist_w = torch.distributions.Normal(ang_w_mean, std)

        pi_values.append(pi_value)

        # entropy_v = dist_v.entropy()
        entropy_w = dist_w.entropy()
        # entropy = entropy_v + entropy_w
        entropy = entropy_w
        entropies.append(entropy)

        if testing:
            # test_std = torch.tensor([0.1])
            d_act_w = ang_w_mean
        else:
            d_act_w = dist_w.sample()

        actions.append(d_act_w)

        w_log_prob = dist_w.log_prob(d_act_w)

        log_prob = w_log_prob
        log_probs.append(log_prob)

        ang_w = st_vel[1] / 270.0

        act_w = ang_w + d_act_w.item()

        if act_w > 1.0:
            act_w = 1.0
        elif act_w < -1.0:
            act_w = -1.0

        state, total_reward, reward, done = process.step(act_w)
        st_vel = state[0]
        states.append(state)
        rewards.append(reward)

        iterations -= 1

    if len(actions) == 1:
        empty_set = True

    return actions, states, torch.tensor(log_probs), rewards, \
           pi_values, v_values, torch.tensor(entropies), total_reward, empty_set


# Отдача
def calculate_discounts(rewards, gamma=0.99):
    discounted_rewards = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)

    return torch.tensor(discounted_rewards)


# GAE
def calculate_gaes(rewards, values, gamma=0.99, decay=0.95):
    next_values = values[1:]
    next_values.append(torch.tensor([0]))
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]
    gaes = torch.zeros(len(rewards))
    gaes[0] = deltas[-1]
    for i in reversed(range(len(deltas) - 1)):
        gaes[len(deltas) - 1 - i] = deltas[i] + decay * gamma * gaes[len(deltas) - 2 - i]
    norm_gaes = (gaes - torch.mean(gaes)) / (torch.std(gaes) + 1e-5)

    return gaes, norm_gaes


def test(process, policy_net, value_net):
    success = 0
    total_reward = 0.0
    test_ep = 1
    for _ in range(test_ep):
        _, _, _, _, _, _, _, reward, empty_set = run_actor(process, policy_net, value_net, testing=True)
        if not empty_set:
            if process.bot.goal_reached:
                success += 1
            total_reward += reward
    success_rate = success / test_ep
    avg_reward = total_reward / test_ep
    return success_rate, success, avg_reward



