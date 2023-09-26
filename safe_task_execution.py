import matplotlib.pyplot as plt
import numpy as np
from numpy import sign
from numpy.linalg import inv
from math import sin, cos, radians, pi, atan2, tan, sqrt, isinf, inf, degrees
from random import randint, uniform, normalvariate, random
from sklearn.cluster import DBSCAN
import threading
import time
import csv
import torch

import RL
from Bot import Bot
from vectorization import getLinesSaM, get_vectorized_obstacles
from map_generation import read_map
from environment import Environment
from lidar_processing import get_lidar_frame, maintain_frame_clustering, get_surrounding_objects, detect_unfamiliar_objects
from visualization import Visualizer
from RL import Value, train, PolicyNew

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Process:
    def __init__(self):
        self.dt = 0.01
        self.bot = Bot(self.dt)
        self.fps = 10
        self.beams_num = 128
        self.map_from_file = read_map('map.csv')
        self.scene = Environment()

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()
        self.cart_lidar_frame = np.zeros([3, self.beams_num], dtype=float)
        self.polar_lidar_frame = np.zeros([3, self.beams_num], dtype=float)
        self.total_reward = 0.0
        self.dist_to_goal = 0.0
        self.prev_dist_to_goal = 0.0
        self.closest = inf
        self.prev_delta_path_dir = 0.0
        self.delta_path_dir = 0.0
        self.is_obst = False
        self.temp_goal_dir = 0.0

    # Состояние процесса
    def get_state(self):
        self.dist_to_goal = sqrt((self.goal_x - self.bot_x) ** 2 + (self.goal_y - self.bot_y) ** 2)
        if self.is_obst:
            self.delta_path_dir = self.temp_goal_dir
        else:
            delta_path_dir = degrees(atan2(self.goal_y - self.bot_y, self.goal_x - self.bot_x)) - self.bot_dir
            if delta_path_dir > 180.0:
                delta_path_dir -= 360.0
            elif delta_path_dir < -180.0:
                delta_path_dir += 360.0
            self.delta_path_dir = delta_path_dir
        state = []

        state.append(np.asarray([self.bot.lin_vel / 4.0, self.bot.ang_vel / 270.0]))
        state.append(np.asarray([self.dist_to_goal, self.delta_path_dir]))
        state.append(np.ones(self.beams_num) - self.polar_lidar_frame[1, :] / 3.0)
        return state

    # Запустить новый эпизод
    def reset(self):
        self.bot_x, self.bot_y = 0.0, 0.0
        self.bot_dir = uniform(-1.0, 1.0) * 90.0
        self.bot.set_position(self.bot_x, self.bot_y, self.bot_dir)

        d = 15
        phi = uniform(-5.0, 5.0)
        self.goal_x, self.goal_y = d * cos(radians(phi)), d * sin(radians(phi))
        self.bot.goal_x = self.goal_x
        self.bot.goal_y = self.goal_y
        self.dist_to_goal = sqrt((self.goal_x - self.bot_x) ** 2 + (self.goal_y - self.bot_y) ** 2)
        self.prev_dist_to_goal = self.dist_to_goal
        delta_path_dir = degrees(atan2(self.goal_y - self.bot_y, self.goal_x - self.bot_x)) - self.bot_dir
        if delta_path_dir > 180.0:
            delta_path_dir -= 360.0
        elif delta_path_dir < -180.0:
            delta_path_dir += 360.0
        self.delta_path_dir = delta_path_dir

        n_obj = int(uniform(3, 5))

        self.scene = Environment()
        ax[0].plot([16, -1, -1, 16, 16], [5, 5, -5, -5, 5], color='white')
        x_offset = 15.0
        for i in range(n_obj):
            s = uniform(0.2, 0.75)
            x = x_offset - uniform(2.5, 3.0)
            y = uniform(-3.0, 3.0)
            lin_vel = uniform(-2.0, 2.0)
            new_obj_nodes = np.asarray([[x - s / 2, x + s / 2, x + s / 2, x - s / 2, x - s / 2],
                                        [y + s / 2, y + s / 2, y - s / 2, y - s / 2, y + s / 2]], dtype=float)
            self.scene.add_object(new_obj_nodes, lin_vel=lin_vel, agent_radius=self.bot.radius, movable=True)
            x_offset = x

        self.cart_lidar_frame, self.polar_lidar_frame = \
            get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)

        # TODO: ВОТ ЗДЕСЬ МЫ ПРОВЕРЯЕМ НАЛИЧИЕ ПРЕПЯТСТВИЯ И ИЗМЕНЯЕМ ЛИНЕЙНУЮ СКОРОСТЬ
        _ = self.check_obstacle()
        self.total_reward = 0.0

        self.bot.terminated = False
        self.bot.goal_reached = False

        # w = degrees(self.bot.LIN_V_MAX * 2 * sin(radians(self.delta_path_dir)) / self.dist_to_goal)

        return self.get_state()

    # Шаг симуляции
    def step(self, ang_vel):
        global bot_img, bot_nose
        reward = 0

        ang_vel *= 270.0

        self.bot.cmd_vel(self.bot.lin_vel, ang_vel)

        if watch:
            goal = [ax[0].scatter([self.goal_x], [self.goal_y], marker='*', s=100, c='green')]

        self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()
        self.dist_to_goal = sqrt((self.goal_x - self.bot_x) ** 2 + (self.goal_y - self.bot_y) ** 2)
        delta_path_dir = degrees(atan2(self.goal_y - self.bot_y, self.goal_x - self.bot_x)) - self.bot_dir
        if delta_path_dir > 180.0:
            delta_path_dir -= 360.0
        elif delta_path_dir < -180.0:
            delta_path_dir += 360.0
        self.delta_path_dir = delta_path_dir

        plots = []
        if self.scene.objects:
            for obj in self.scene.objects:
                if obj.movable:
                    plot, = ax[0].plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='red')
                    plots.append(plot)
                    obj.transform(1 / self.fps)
                    if obj.check_agent_collision(self.bot):
                        break
                else:
                    ax[0].plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='black', linewidth=4)

        if self.bot.terminated:
            reward -= 25
            if plots:
                for plot in plots:
                    plot.remove()
            goal[0].remove()
            self.total_reward += reward
            return self.get_state(), self.total_reward, reward, True

        if watch:
            bot_img = plt.Circle((self.bot_x, self.bot_y), self.bot.radius, color='gray')
            bot_nose = plt.Rectangle((self.bot_x + 0.01 * sin(radians(self.bot_dir)),
                                      self.bot_y - 0.01 * sin(radians(self.bot_dir))),
                                     self.bot.radius, 0.02,
                                     angle=self.bot_dir, rotation_point='xy', color='black')

            ax[0].add_patch(bot_img)
            ax[0].add_patch(bot_nose)

            ax[1].clear()

        self.cart_lidar_frame, self.polar_lidar_frame = \
            get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)

        # TODO: ВОТ ЗДЕСЬ МЫ ПРОВЕРЯЕМ НАЛИЧИЕ ПРЕПЯТСТВИЯ И ИЗМЕНЯЕМ ЛИНЕЙНУЮ СКОРОСТЬ
        count_pts = self.check_obstacle()
        reward -= 0.01 * count_pts

        current_frame = np.ones(self.beams_num) - self.polar_lidar_frame[1, :] / 5

        if watch:
            # ось лидара
            ax[1].scatter([0.0], [0.0], s=10, color='black')

            # серые точки кадра
            ax[1].scatter(self.cart_lidar_frame[0, ::2], self.cart_lidar_frame[1, ::2], s=4, marker='o', color='gray')

        if watch:
            plt.draw()
            plt.pause(1 / self.fps)

            if plots:
                for plot in plots:
                    plot.remove()

            bot_img.remove()
            bot_nose.remove()
            goal[0].remove()

        else:
            time.sleep(1/self.fps)

        if not self.bot.goal_reached:
            delta_dist = self.prev_dist_to_goal - self.dist_to_goal
            if delta_dist > 0:
                k_d = 2.5
            else:
                k_d = 4.0
            reward += k_d * delta_dist  # награда/штраф за изменение расстояния до цели

            reward -= 1.0 - self.bot.lin_vel / self.bot.LIN_V_MAX

            reward -= 0.01 * abs(self.delta_path_dir)

            close = (current_frame[:] > 0.8).sum()
            reward -= 0.05 * close

            if self.bot.terminated:
                reward -= 25.0  # штраф за столкновение
        else:
            reward += 25.0  # награда за достижение цели

        self.total_reward += reward

        self.prev_dist_to_goal = self.dist_to_goal

        return self.get_state(), self.total_reward, reward, self.bot.terminated or self.bot.goal_reached

    # Проверка препятствий по направлению взгляда агента
    def check_obstacle(self):
        closest_front = 2.5
        closest_front_ang = 0.0
        closest_total = 3.0
        closest_total_ang = 0.0
        count = 0
        self.is_obst = False
        is_front = False
        for ang, dist in zip(self.polar_lidar_frame[0], self.polar_lidar_frame[1]):
            if dist < 3.0:
                self.is_obst = True
                if abs(dist * cos(ang)) < 0.5:
                    count += 1
                    d = dist * sin(ang) - 0.5
                    if d < closest_front:
                        closest_front = d
                        closest_front_ang = ang
                        is_front = True
                if dist < closest_total:
                    closest_total_ang = ang

        if abs(degrees(closest_front_ang) - 90.0) > 75.0:
            lin_vel = self.bot.LIN_V_MAX
        else:
            lin_vel = (closest_front / 2.5) * self.bot.LIN_V_MAX
        self.bot.lin_vel = lin_vel

        if self.is_obst:
            if is_front:
                delta_path_dir = degrees(atan2(self.goal_y - self.bot_y, self.goal_x - self.bot_x)) - self.bot_dir
                if delta_path_dir > 180.0:
                    delta_path_dir -= 360.0
                elif delta_path_dir < -180.0:
                    delta_path_dir += 360.0
                self.temp_goal_dir = 90.0 * sign(cos(closest_total_ang))
                sm = delta_path_dir - self.temp_goal_dir
                if sm > 120.0 or sm < -120.0:
                    self.temp_goal_dir *= -1

                # TODO: надо рассмотреть краевые углы препятствия перед носом.
                #  Также стоит учесть направление к цели /

            else:
                self.temp_goal_dir = self.delta_path_dir + degrees(closest_total_ang) + 90.0 * sign(cos(closest_total_ang)) - 90.0

        return count





fig, ax = plt.subplots(1, 2)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

process = Process()


x0, y0, dir0 = process.bot.get_current_position()
bot_img = plt.Circle((x0, y0), process.bot.radius, color='gray')
bot_nose = plt.Rectangle((x0 + 0.01 * sin(radians(dir0)),
                          y0 - 0.01 * sin(radians(dir0))),
                         process.bot.radius, 0.02,
                         angle=dir0, rotation_point='xy', color='black')

watch = True

# policy_net = Policy(3)
value_net = Value(3)
policy_net = PolicyNew(128)

policy_states = torch.load(r'C:\Users\User\PycharmProjects\botMaster\Best\var_dirs60-60_dist3-5.pth')
policy_net.load_state_dict(policy_states)

e_p = 3  # число эпох для оптимизации политики П
e_v = 3  # число эпох для оптимизации полезности V
e_d = 2  # число эпох дистилляции
total_episodes = 100
test_period = 1

avg_rewards = []
checkpoints = []
loss = []

replay = False

count_replay = 0

for ep in range(1, total_episodes+1):

    # Ранее вспомогательный в процессе обучения код

    # torch.save(policy_net.state_dict(),
    #            r'C:\Users\User\PycharmProjects\botMaster\PPP\P_S_' + '_EP_' + str(ep) + '.pth')
    # train_reward = RL.train(process, policy_net, value_net, e_p, e_v, e_d)
    # avg_rewards.append(train_reward)
    # checkpoints.append(ep)


    if ep % test_period == 0:
        print("TESTING")
        success_rate, success, avg_reward = RL.test(process, policy_net, value_net)
        avg_rewards.append(avg_reward)
        checkpoints.append(ep)
        print('Success', success_rate, 'at episode', ep, 'with average reward', avg_reward)

        # Ранее вспомогательный в процессе тестирования код

        # print('Episode', ep, 'with average reward', avg_reward)
        # torch.save(policy_net.state_dict(),
        #            r'C:\Users\User\PycharmProjects\botMaster\PPP\P_S_' + str(success) + '_EP_' + str(ep) + '.pth')
        #
        # # if avg_reward < -50.0:
        # #     replay = True
        # #     if count_replay == max_replay:
        # #         replay = False
        # #         count_replay = 0
        # #     else:
        # #         count_replay += 1
        # #         print("REPLAYING", count_replay)
        # # else:
        # #     replay = False
        # #     torch.save(policy_net.state_dict(),
        # #                r'C:\Users\User\PycharmProjects\botMaster\PPP\P_S_' + '_EP_' + str(ep) + '.pth')
        # #     count_replay = 0
        #
        # torch.save(policy_net.state_dict(),
        #            r'C:\Users\User\PycharmProjects\botMaster\PPP\P_S_' + str(success) + '_EP_' + str(ep) + '.pth')



# Не используемый в последствии код

# _, r_ax = plt.subplots()
# r_ax.plot(checkpoints, avg_rewards)
# plt.show()


# # Движение по карте с незнакомыми препятствиями
# map_from_file = read_map('map.csv')
# beams_num = 300
# # motion_lin_vel = 1
# # sim_lin_vel = 2 * motion_lin_vel
#
# # TODO: при возникновении тупняков на визуализации подобрать параметры
# fps = 20  # ЭТО СТРОГО
# discr_dt = 0.01  # 0.01 - 0.1 норм?
#
# # bot = Bot(discr_dt)


# determined_policy = PolicyNew(128)
# policy_states = torch.load(r'C:\Users\User\PycharmProjects\botMaster\Best\var_dirs60-60_dist3-5.pth')
# determined_policy.load_state_dict(policy_states)
#
# w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6 = determined_policy.get_params()
#
# policy_net.set_param(w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6)


# policy_states = torch.load(r'C:\Users\User\PycharmProjects\botMaster\Best\var_dirs60-60_dist3-5.pth')
# policy_states = torch.load(r'C:\Users\User\PycharmProjects\botMaster\Best\straight_5_05.pth')
# policy_states = torch.load(r'C:\Users\User\PchyarmProjects\botMaster\Best\var_dirs_5_05.pth')
# policy_states = torch.load(r'C:\Users\User\PycharmProjects\botMaster\Best\var_dirs60-60_dist3-5.pth')

# policy_net.load_state_dict(policy_states)

# value_states = torch.load(r'C:\Users\User\PycharmProjects\botMaster\VVV\V_S_23_EP_80.pth')
# value_net.load_state_dict(value_states)

# print(RL.test(process, policy_net, value_net))
