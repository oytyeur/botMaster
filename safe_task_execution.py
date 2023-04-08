import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from math import sin, cos, radians, pi, atan2, tan, sqrt, isinf, inf, degrees
from random import randint, uniform, normalvariate, random
from sklearn.cluster import DBSCAN
import threading
import time
import csv
import torch

from Bot import Bot
from vectorization import getLinesSaM, get_vectorized_obstacles
from map_generation import read_map
from environment import Environment
from lidar_processing import get_lidar_frame, maintain_frame_clustering, get_surrounding_objects, detect_unfamiliar_objects
from visualization import Visualizer
from RL import Policy, train


class Process:
    def __init__(self):
        self.dt = 0.01
        self.bot = Bot(self.dt)
        self.fps = 30
        self.beams_num = 103
        self.map_from_file = read_map('map.csv')
        self.scene = Environment()

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()
        self.cart_lidar_frame = np.zeros([3, self.beams_num], dtype=float)
        self.polar_lidar_frame = np.zeros([3, self.beams_num], dtype=float)
        self.obst_marked_lidar_frame = -np.ones([self.beams_num], dtype=float)
        # self.state = np.zeros([1, self.beams_num+7], dtype=float)
        self.reward = 0.0
        # self.done = not (self.bot.goal_reached or self.bot.terminated)
        self.dist_to_goal = 0.0
        self.closest = inf
        self.delta_path_dir = 0.0

    # Состояние процесса
    def get_state(self):
        state = np.asarray([self.bot.lin_vel, self.bot.ang_vel,
                            self.bot_x, self.bot_y, self.bot_dir,
                            self.goal_x, self.goal_y], #self.dist_to_goal, self.delta_path_dir],
                            dtype=float)
        state = np.append(state, self.polar_lidar_frame[1, ::3])  # срез - в нейросеть идёт каждый второй
        state = np.append(state, self.obst_marked_lidar_frame[::3])  # срез меток кадра
        self.obst_marked_lidar_frame = -np.ones([self.beams_num], dtype=float)
        return state

    # # Обновление значения награды
    # def update_reward(self):
    #     if not

    # # Запустить новый эпизод
    def reset(self):
        self.bot_x = uniform(-0.5, 0)
        self.bot_y = uniform(-0.5, 0.5)
        self.bot_dir = uniform(-30.0, 30.0)
        self.bot.set_position(self.bot_x, self.bot_y, self.bot_dir)
        self.goal_x = uniform(9.25, 9.75)
        self.goal_y = uniform(-2.0, 2.0)
        self.bot.goal_x = self.goal_x
        self.bot.goal_y = self.goal_y
        self.dist_to_goal = sqrt((self.goal_x - self.bot_x) ** 2 + (self.goal_y - self.bot_y) ** 2)
        self.closest = inf
        self.delta_path_dir = degrees(atan2(self.goal_y - self.bot_y, self.goal_x - self.bot_x)) - self.bot_dir
        n_obj = int(uniform(4, 5))
        self.scene = Environment()
        ax[0].plot([10.5, -1.5, -1.5, 10.5, 10.5], [3.5, 3.5, -3.5, -3.5, 3.5], color='white')
        x_offset = 8.0
        # for i in range(n_obj):
        #     s = uniform(0.3, 0.7)
        #     x = x_offset - uniform(0.75, 1)
        #     y = uniform(-2.5, 2.5)
        #     lin_vel = uniform(-2.0, 2.0)
        #     new_obj_nodes = np.asarray([[x - s / 2, x + s / 2, x + s / 2, x - s / 2, x - s / 2],
        #                                 [y + s / 2, y + s / 2, y - s / 2, y - s / 2, y + s / 2]], dtype=float)
        #     self.scene.add_object(new_obj_nodes, lin_vel=lin_vel, agent_radius=self.bot.radius, movable=True)
        #     x_offset = x

        self.cart_lidar_frame, self.polar_lidar_frame = \
            get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)

        # self.state = np.zeros([1, self.beams_num+7], dtype=float)
        self.reward = 0.0

        self.bot.terminated = False
        self.bot.goal_reached = False

        w = degrees(self.bot.LIN_V_MAX * 2 * sin(radians(self.delta_path_dir)) / self.dist_to_goal)
        self.bot.cmd_vel(self.bot.LIN_V_MAX, w)

        return self.get_state()

    # Шаг симуляции
    def step(self, lin_vel, ang_vel):
        global bot_img, bot_nose

        self.bot.cmd_vel(lin_vel, ang_vel)

        time.sleep(1/self.fps)

        goal = [ax[0].scatter([self.goal_x], [self.goal_y], marker='*', s=100, c='green')]

        # self.bot_x, self.bot_y, self.bot_dir = self.bot.goal_reached_check(self.goal_x, self.goal_y, threshold=0.05)
        self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()
        # if self.bot.goal_reached:
        #     self.reward += 10000
        #     return self.get_state(), self.reward, True

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

        # if self.bot.terminated:
        #     # self.reward -= 100
        #     if plots:
        #         for plot in plots:
        #             plot.remove()
        #     goal[0].remove()
        #     return self.get_state(), self.reward, True

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

        obstacles, obst_idxs, vect_obstacles = self.analyze()
        for obst in obst_idxs:
            self.obst_marked_lidar_frame[obst] = 1.0

        # ось лидара
        ax[1].scatter([0.0], [0.0], s=10, color='black')

        # серые точки кадра
        ax[1].scatter(self.cart_lidar_frame[0, ::2], self.cart_lidar_frame[1, ::2], s=4, marker='o', color='gray')

        # # цветные точки после кластеризации
        # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

        if obst_idxs:
            for obst in obst_idxs:
                ax[1].scatter(self.cart_lidar_frame[0, obst], self.cart_lidar_frame[1, obst],
                              s=4, marker='o', color='red')

        if vect_obstacles:
            for obst in vect_obstacles:
                for i in range(obst.shape[1] - 1):
                    ax[1].plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
                                  linewidth=1.5, color='magenta')

        plt.draw()
        plt.pause(1 / self.fps)

        if plots:
            for plot in plots:
                plot.remove()

        bot_img.remove()
        bot_nose.remove()
        goal[0].remove()

        if not self.bot.goal_reached:
            self.delta_path_dir = abs(degrees(atan2(self.goal_y - self.bot_y, self.goal_x - self.bot_x)) - self.bot_dir)
            self.dist_to_goal = sqrt((self.goal_x - self.bot_x) ** 2 + (self.goal_y - self.bot_y) ** 2)
            if self.dist_to_goal < self.closest:
                self.closest = self.dist_to_goal

            self.reward -= self.dist_to_goal ** 2
            if self.bot.terminated:
                if self.closest < 2:
                    if self.closest < 1:
                        self.reward += (1 - self.closest) * 100000
                    else:
                        self.reward += (((2 - self.closest) ** 2) * 100000)
                else:
                    self.reward -= 100000
                # self.reward -= (10000 * self.dist_to_goal ** 2 + 10000)

        else:
            self.reward += 200000


        return self.get_state(), self.reward, self.bot.terminated or self.bot.goal_reached



    # Движение из точки в точку
    def p2p_motion(self, lin_vel):
        global bot_img, bot_nose
        self.bot.aligned = False

        goal = [ax[0].scatter([self.goal_x], [self.goal_y], marker='*', s=100, c='green')]

        while not self.bot.goal_reached:
            self.bot_x, self.bot_y, self.bot_dir = \
                self.bot.move_to_pnt_check(self.goal_x, self.goal_y, lin_vel, self.fps)

            plots = []
            for obj in self.scene.objects:
                if not obj.movable:
                    ax[0].plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='black', linewidth=4)
                else:
                    plot, = ax[0].plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='red')
                    plots.append(plot)
                    obj.transform(1 / self.fps)
                    if obj.check_agent_collision(self.bot):
                        break

            if self.bot.terminated:
                if plots:
                    for plot in plots:
                        plot.remove()
                break

            bot_img = plt.Circle((self.bot_x, self.bot_y), self.bot.radius, color='grey')
            bot_nose = plt.Rectangle((self.bot_x + 0.01 * sin(radians(self.bot_dir)),
                                      self.bot_y - 0.01 * sin(radians(self.bot_dir))),
                                     self.bot.radius, 0.02,
                                     angle=self.bot_dir, rotation_point='xy', color='black')

            ax[0].add_patch(bot_img)
            ax[0].add_patch(bot_nose)

            ax[1].clear()

            self.cart_lidar_frame, self.polar_lidar_frame = \
                get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)

            # # лучи лидара
            # for i in range(frame.shape[1]):
            #     lidar_ax.plot([0.0, frame[0, i]], [0.0, frame[1, i]],
            #                   linewidth=0.1, color='red')

            obstacles, obst_idxs, vect_obstacles = self.analyze()

            # ось лидара
            ax[1].scatter([0.0], [0.0], s=10, color='red')

            # серые точки кадра
            ax[1].scatter(self.cart_lidar_frame[0, ::3], self.cart_lidar_frame[1, ::3], s=4, marker='o', color='gray')

            # # цветные точки после кластеризации
            # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

            if obst_idxs:
                for obst in obst_idxs:
                    ax[1].scatter(self.cart_lidar_frame[0, obst], self.cart_lidar_frame[1, obst],
                                  s=4, marker='o', color='red')

            if vect_obstacles:
                for obst in vect_obstacles:
                    for i in range(obst.shape[1]-1):
                        ax[1].plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
                                    linewidth=1.5, color='magenta')

            plt.draw()
            plt.pause(1/self.fps)

            if plots:
                for plot in plots:
                    plot.remove()

            bot_img.remove()
            bot_nose.remove()

        goal[0].remove()
        self.bot.goal_reached = False

    # # Движение по команде скорости
    # def execute_cmd_vel(self, lin_vel, ang_vel):
    #     global bot_img, bot_nose
    #
    #     self.bot.cmd_vel(lin_vel, ang_vel)
    #
    #     goal = [ax.scatter([self.goal_x], [self.goal_y], marker='*', s=100, c='green')]
    #
    #     t0 = time.time()
    #     while time.time() - t0 < 1:
    #         self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()
    #         # ax.scatter([c_x], [c_y], s=0.1, c='cyan')
    #
    #         plots = []
    #         for obj in self.scene.objects:
    #             if not obj.movable:
    #                 ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='black', linewidth=4)
    #             else:
    #                 plot, = ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='grey')
    #                 plots.append(plot)
    #                 obj.transform(1 / self.fps)
    #                 if obj.check_agent_collision(self.bot):
    #                     break
    #
    #         if self.bot.terminated:
    #             if plots:
    #                 for plot in plots:
    #                     plot.remove()
    #             break
    #
    #         bot_img = plt.Circle((self.bot_x, self.bot_y), self.bot.radius, color='r')
    #         bot_nose = plt.Rectangle((self.bot_x + 0.01 * sin(radians(self.bot_dir)),
    #                                   self.bot_y - 0.01 * sin(radians(self.bot_dir))),
    #                                  self.bot.radius, 0.02,
    #                                  angle=self.bot_dir, rotation_point='xy', color='black')
    #
    #         ax.add_patch(bot_img)
    #         ax.add_patch(bot_nose)
    #
    #         lidar_ax.clear()
    #
    #         self.cart_lidar_frame, self.polar_lidar_frame = \
    #             get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)
    #
    #         vect_obstacles = self.analyze()
    #
    #         # ось лидара
    #         lidar_ax.scatter([0.0], [0.0], s=10, color='red')
    #
    #         # серые точки кадра
    #         lidar_ax.scatter(self.cart_lidar_frame[0, :], self.cart_lidar_frame[1, :], s=4, marker='o', color='gray')
    #
    #         # # цветные точки после кластеризации
    #         # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')
    #
    #         if vect_obstacles:
    #             for obst in vect_obstacles:
    #                 for i in range(obst.shape[1] - 1):
    #                     lidar_ax.plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
    #                                   linewidth=1.5, color='magenta')
    #
    #         plt.draw()
    #         plt.pause(1/self.fps)
    #
    #         if plots:
    #             for plot in plots:
    #                 plot.remove()
    #
    #         bot_img.remove()
    #         bot_nose.remove()
    #
    #     goal[0].remove()

    def analyze(self):
        clustered = maintain_frame_clustering(self.cart_lidar_frame, eps=0.4)
        objects, obj_idxs, idxs_lst = get_surrounding_objects(self.cart_lidar_frame, clustered)
        # objects - отдельные объекты в виде декартовых координат
        # obj_idxs - срезы индексов на кадре для каждого отдельного объекта, в т.ч. шумов
        # idxs_lst - индексы элементов списка idxs, не являющихся шумом (индексы объектов, соответствующих objects)
        obstacles, obst_idxs = detect_unfamiliar_objects(self.map_from_file, self.bot_x, self.bot_y, self.bot_dir,
                                                        objects, obj_idxs, idxs_lst, threshold=0.1)
        # vect_obstacles = get_vectorized_obstacles(obstacles)

        vect_obstacles = []

        return obstacles, obst_idxs, vect_obstacles

    def wait_for_assignment(self):
        global bot_img, bot_nose
        self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()

        print('Waiting for assignment')

        ax[0].add_patch(bot_img)
        ax[0].add_patch(bot_nose)

        assigned = False

        t0 = time.time()

        while not assigned:
            plots = []
            for obj in self.scene.objects:
                if not obj.movable:
                    ax[0].plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='black', linewidth=4)
                else:
                    plot, = ax[0].plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='red')
                    plots.append(plot)
                    obj.transform(1 / self.fps)

            ax[1].clear()
            self.cart_lidar_frame, self.polar_lidar_frame = \
                get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)

            # лучи лидара
            # for i in range(cart_lidar_frame.shape[1]):
            #     lidar_ax.plot([0.0, cart_lidar_frame[0, i]], [0.0, cart_lidar_frame[1, i]],
            #                   linewidth=0.1, color='red')

            obstacles, obst_idxs, vect_obstacles = self.analyze()

            # ось лидара
            ax[1].scatter([0.0], [0.0], s=10, color='red')

            # серые точки кадра
            ax[1].scatter(self.cart_lidar_frame[0, :], self.cart_lidar_frame[1, :], s=4, marker='o', color='gray')

            # # цветные точки после кластеризации
            # lidar_ax.scatter(cart_lidar_frame[0, :], cart_lidar_frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

            if obst_idxs:
                for obst in obst_idxs:
                    ax[1].scatter(self.cart_lidar_frame[0, obst], self.cart_lidar_frame[1, obst],
                                  s=4, marker='o', color='red')

            if vect_obstacles:
                for obst in vect_obstacles:
                    for i in range(obst.shape[1] - 1):
                        ax[1].plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
                                      linewidth=1.5, color='magenta')

            plt.draw()
            plt.pause(1 / self.fps)

            if plots:
                for plot in plots:
                    plot.remove()

            if time.time() - t0 > 0.1:
                assigned = True
                ax[0].clear()

            # time.sleep(1/fps)

    def execute_random_task(self):
        self.reset()
        v = 2
        d = self.dist_to_goal
        d_dir = self.delta_path_dir
        w = degrees(v * 2 * sin(radians(d_dir)) / d)
        # cmd_vel task
        c = 0
        for j in range(120):
            self.step(v, w)
            c += 1
            if self.bot.terminated or self.bot.goal_reached:
                break
        print(c)

        # self.p2p_motion(2.0)

        self.wait_for_assignment()
        self.bot.terminated = False


# vis = Visualizer()
# vis = Visualizer(see_scene=False, see_lidar=False)

# scene = Environment()
# new_object = np.asarray([[-0.25, 0.25, 0.25, -0.25, -0.25], [2.25, 2.25, 1.75, 1.75, 2.25]], dtype=float)
# scene.add_object(new_object, lin_vel=-1.0, ang_vel=0.5, dir=-pi, movable=True)
#
# new_object = np.asarray([[-0.25, 0.25, 0.25, -0.25, -0.25], [-2.25, -2.25, -1.75, -1.75, -2.25]], dtype=float)
# scene.add_object(new_object, lin_vel=1.0, ang_vel=0.5, dir=0.0, movable=True)

fig, ax = plt.subplots(1, 2)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')

process = Process()


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


x0, y0, dir0 = process.bot.get_current_position()
bot_img = plt.Circle((x0, y0), process.bot.radius, color='gray')
bot_nose = plt.Rectangle((x0 + 0.01 * sin(radians(dir0)),
                          y0 - 0.01 * sin(radians(dir0))),
                         process.bot.radius, 0.02,
                         angle=dir0, rotation_point='xy', color='black')


# policy = Policy(77, 2)
policy = torch.load('policy.pkl')
policy.eval()
device = torch.device('cuda')

# policy.to(device)

rewards = []
for i in range(100):
    episode_reward = train(process, policy)
    rewards.append(episode_reward)
    print(f'Episode {i}: Reward {episode_reward}')
    if not i % 100 and i > 0:
        torch.save(policy, 'waypoint_policy.pkl')
        print('Checkpoint')

# policy.to('cpu')
torch.save(policy, 'policy.pkl')

# for ep in range(50):
#     process.execute_random_task()

# for i in range(len(rewards) - 1):
#     r_ax.plot([rewards[i], rewards[i+1]])
_, r_ax = plt.subplots()
r_ax.plot(rewards)

# # p2p_motion(0, 0, 0, sim_lin_vel, scene, bot, fps, beams_num=beams_num)
#
#
# p2p_motion(4, 4, 0, sim_lin_vel, scene, bot, fps, beams_num=beams_num)
# wait_for_assignment(bot)






# ==================================== ПРОИЗВОЛЬНАЯ КАРТА ===========================================


# # ДВИЖЕНИЕ С ИМЕЮЩЕЙСЯ КАРТОЙ, ОТРАБОТКА ТРАЕКТОРИИ
# # Добавление незакартированных препятствий
# scene_contours.append(np.asarray([[-4, -4, -3, -3, -4], [4, 3, 3, 4, 4]], dtype=float))
# scene_contours.append(np.asarray([[0, -0.5, -0.5, 0, 0], [4, 4, 3.5, 3.5, 4]], dtype=float))
# scene_contours.append(np.asarray([[-4, -4.5, -4.5, -4, -4], [0, 0, 0.5, 0.5, 0]], dtype=float))
# scene_contours.append(np.asarray([[4.4, 4.9, 4.9, 4.4, 4.4], [3, 3, 4, 4, 3]], dtype=float))
# scene_contours.append(np.asarray([[4.4, 4.9, 4.9, 4.4, 4.4], [-3, -3, -4, -4, -3]], dtype=float))
# scene_contours.append(np.asarray([[4, 3.5, 3.5, 4, 4], [-4, -4, -4.5, -4.5, -4]], dtype=float))
# scene_contours.append(np.asarray([[-1, 0, 0.5, -1], [-4.7, -4.3, -4.5, -4.7]], dtype=float))


# # Движение по карте с незнакомыми препятствиями
# map_from_file = read_map('map.csv')
# beams_num = 300
# motion_lin_vel = 2
# sim_lin_vel = 2 * motion_lin_vel

# p2p_motion(4, 0, 0, motion_lin_vel, fps, beams_num=beams_num, mapping=False)

# p2p_motion(-2, 4, 0, sim_lin_vel, scene, bot, fps, beams_num=beams_num)
# p2p_motion(-3, -4.2, 0, sim_lin_vel, scene, bot, fps, beams_num=beams_num)
# p2p_motion(4, -3, 0, sim_lin_vel, scene, bot, fps, beams_num=beams_num)
# p2p_motion(3.8, 0, 0, sim_lin_vel, scene, bot, fps, beams_num=beams_num)
# p2p_motion(0, 0, 0, sim_lin_vel, scene, bot, fps, beams_num=beams_num)
# wait_for_assignment(bot)



# ПОКАЗАТЬ КАДР В НЕКОТОРОЙ ПОЗИЦИИ РОБОТА
# bot.x = 3
# bot.y = 1
# bot.dir = -90
# get_single_frame()



# # map_contours = get_map_contours(map_from_file, map_clustered)
# # for mp_cnt in map_contours:
# #     nodes = np.zeros([2, mp_cnt.shape[1]])
# #     Nlines = getLines(nodes, mp_cnt, mp_cnt.shape[1], tolerance=0.1)
# #     for i in range(Nlines):
# #         map_ax.plot([nodes[0, i], nodes[0, i + 1]], [nodes[1, i], nodes[1, i + 1]], linewidth=2)

# # # # Это больше для работы с ROI нужно было
# # # # for fr in map:
# # # #     map_ax.scatter(fr[0, :], fr[1, :], s=1, marker='o', color='gray')
# # # map_ax.scatter(map_from_file[0, :], map_from_file[1, :], s=1, marker='o', color='gray')
# # # # for obst in obstacles:
# # # #     map_ax.scatter(obst[0, :], obst[1, :], s=5, marker='o', color='red')
# # # i = 2
# # # map_ax.scatter(obstacles[i][0, :], obstacles[i][1, :], s=5, marker='o', color='red')
# # # # map_ax.scatter(objects[0, :], objects[1, :], s=1, c=clustered.labels_, cmap='tab10')

plt.show()
