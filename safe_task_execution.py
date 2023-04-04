import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from math import sin, cos, radians, pi, atan2, tan, sqrt, isinf, inf
from random import randint, uniform, normalvariate, random
from sklearn.cluster import DBSCAN
import threading
import time
import csv

from Bot import Bot
from vectorization import getLinesSaM, get_vectorized_obstacles
from map_generation import read_map
from environment import Environment
from lidar_processing import get_lidar_frame, maintain_frame_clustering, get_surrounding_objects, detect_unfamiliar_objects
from visualization import Visualizer


class Process:
    def __init__(self):
        self.dt = 0.01
        self.bot = Bot(self.dt)
        self.fps = 20
        self.beams_num = 300
        self.map_from_file = read_map('map.csv')
        self.scene = Environment()

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()
        self.cart_lidar_frame = np.zeros([3, self.beams_num])
        self.polar_lidar_frame = np.zeros([3, self.beams_num])
        self.state = np.zeros([1, 307], dtype=float)
        self.reward = 0.0
        # self.done = not (self.bot.goal_reached or self.bot.terminated)

    # Состояние процесса
    def get_state(self):
        state = np.asarray([x0, y0, dir0, self.goal_x, self.goal_y, self.bot.lin_vel, self.bot.ang_vel],
                           dtype=float)
        state = np.append(state, self.polar_lidar_frame[1, :])

        return state.T

    # # Обновление значения награды
    # def update_reward(self):
    #     if not
    #
    # # # Шаг симуляции
    # def step(self, lin_vel, ang_vel):
    #
    #     if not self.bot.goal_reached:
    #         self.reward -= 1

    # # Запустить новый эпизод
    def reset(self):
        self.bot.set_position(uniform(-0.5, 0), uniform(-2.0, 2.0), uniform(-45.0, 45.0))
        self.goal_x = uniform(9.25, 9.75)
        self.goal_y = uniform(-2.0, 2.0)
        n_obj = int(uniform(5, 8))
        self.scene = Environment()
        ax.plot([10.5, -1.5, -1.5, 10.5, 10.5], [3.5, 3.5, -3.5, -3.5, 3.5], color='white')
        x_offset = 2.0
        for i in range(n_obj):
            s = uniform(0.3, 0.7)
            x = x_offset + uniform(0.75, 1)
            y = uniform(-2.5, 2.5)
            lin_vel = uniform(-4.0, 4.0)
            new_obj_nodes = np.asarray([[x - s / 2, x + s / 2, x + s / 2, x - s / 2, x - s / 2],
                                        [y + s / 2, y + s / 2, y - s / 2, y - s / 2, y + s / 2]], dtype=float)
            self.scene.add_object(new_obj_nodes, lin_vel=lin_vel, agent_radius=self.bot.radius, movable=True)
            x_offset = x

        return self.get_state()

    # Движение из точки в точку
    def p2p_motion(self, lin_vel):
        global bot_img, bot_nose
        self.bot.aligned = False

        goal = [ax.scatter([self.goal_x], [self.goal_y], marker='*', s=100, c='green')]

        while not self.bot.goal_reached:
            self.bot_x, self.bot_y, self.bot_dir = \
                self.bot.move_to_pnt_check(self.goal_x, self.goal_y, lin_vel, self.fps)

            plots = []
            for obj in self.scene.objects:
                if not obj.movable:
                    ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='black', linewidth=4)
                else:
                    plot, = ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='grey')
                    plots.append(plot)
                    obj.transform(1 / self.fps)
                    if obj.check_agent_collision(self.bot):
                        break

            if self.bot.terminated:
                if plots:
                    for plot in plots:
                        plot.remove()
                break

            bot_img = plt.Circle((self.bot_x, self.bot_y), self.bot.radius, color='r')
            bot_nose = plt.Rectangle((self.bot_x + 0.01 * sin(radians(self.bot_dir)),
                                      self.bot_y - 0.01 * sin(radians(self.bot_dir))),
                                     self.bot.radius, 0.02,
                                     angle=self.bot_dir, rotation_point='xy', color='black')

            ax.add_patch(bot_img)
            ax.add_patch(bot_nose)

            lidar_ax.clear()

            self.cart_lidar_frame, self.polar_lidar_frame = \
                get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)

            # # лучи лидара
            # for i in range(frame.shape[1]):
            #     lidar_ax.plot([0.0, frame[0, i]], [0.0, frame[1, i]],
            #                   linewidth=0.1, color='red')

            vect_obstacles = self.analyze()

            # ось лидара
            lidar_ax.scatter([0.0], [0.0], s=10, color='red')

            # серые точки кадра
            lidar_ax.scatter(self.cart_lidar_frame[0, :], self.cart_lidar_frame[1, :], s=4, marker='o', color='gray')

            # # цветные точки после кластеризации
            # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

            if vect_obstacles:
                for obst in vect_obstacles:
                    for i in range(obst.shape[1]-1):
                        lidar_ax.plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
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

    # Движение по команде скорости
    def execute_cmd_vel(self, lin_vel, ang_vel):
        global bot_img, bot_nose

        self.bot.cmd_vel(lin_vel, ang_vel)

        goal = [ax.scatter([self.goal_x], [self.goal_y], marker='*', s=100, c='green')]

        t0 = time.time()
        while time.time() - t0 < 1:
            self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()
            # ax.scatter([c_x], [c_y], s=0.1, c='cyan')

            plots = []
            for obj in self.scene.objects:
                if not obj.movable:
                    ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='black', linewidth=4)
                else:
                    plot, = ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='grey')
                    plots.append(plot)
                    obj.transform(1 / self.fps)
                    if obj.check_agent_collision(self.bot):
                        break

            if self.bot.terminated:
                if plots:
                    for plot in plots:
                        plot.remove()
                break

            bot_img = plt.Circle((self.bot_x, self.bot_y), self.bot.radius, color='r')
            bot_nose = plt.Rectangle((self.bot_x + 0.01 * sin(radians(self.bot_dir)),
                                      self.bot_y - 0.01 * sin(radians(self.bot_dir))),
                                     self.bot.radius, 0.02,
                                     angle=self.bot_dir, rotation_point='xy', color='black')

            ax.add_patch(bot_img)
            ax.add_patch(bot_nose)

            lidar_ax.clear()

            self.cart_lidar_frame, self.polar_lidar_frame = \
                get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)

            vect_obstacles = self.analyze()

            # ось лидара
            lidar_ax.scatter([0.0], [0.0], s=10, color='red')

            # серые точки кадра
            lidar_ax.scatter(self.cart_lidar_frame[0, :], self.cart_lidar_frame[1, :], s=4, marker='o', color='gray')

            # # цветные точки после кластеризации
            # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

            if vect_obstacles:
                for obst in vect_obstacles:
                    for i in range(obst.shape[1] - 1):
                        lidar_ax.plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
                                      linewidth=1.5, color='magenta')

            plt.draw()
            plt.pause(1/self.fps)

            if plots:
                for plot in plots:
                    plot.remove()

            bot_img.remove()
            bot_nose.remove()

        goal[0].remove()

    def analyze(self):
        clustered = maintain_frame_clustering(self.cart_lidar_frame, eps=0.4)
        objects = get_surrounding_objects(self.cart_lidar_frame, clustered)
        obstacles = detect_unfamiliar_objects(self.map_from_file, self.bot_x, self.bot_y, self.bot_dir,
                                              objects, threshold=0.1)
        vect_obstacles = get_vectorized_obstacles(obstacles)

        return vect_obstacles

    def wait_for_assignment(self):
        global bot_img, bot_nose
        self.bot_x, self.bot_y, self.bot_dir = self.bot.get_current_position()

        print('Waiting for assignment')

        ax.add_patch(bot_img)
        ax.add_patch(bot_nose)

        assigned = False
        t0 = time.time()

        while not assigned:
            plots = []
            for obj in self.scene.objects:
                if not obj.movable:
                    ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='black', linewidth=4)
                else:
                    plot, = ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='grey')
                    plots.append(plot)
                    obj.transform(1 / self.fps)

            lidar_ax.clear()
            self.cart_lidar_frame, self.polar_lidar_frame = \
                get_lidar_frame(self.bot_x, self.bot_y, self.bot_dir, self.scene.objects, self.beams_num)

            # лучи лидара
            # for i in range(cart_lidar_frame.shape[1]):
            #     lidar_ax.plot([0.0, cart_lidar_frame[0, i]], [0.0, cart_lidar_frame[1, i]],
            #                   linewidth=0.1, color='red')

            vect_obstacles = self.analyze()

            # ось лидара
            lidar_ax.scatter([0.0], [0.0], s=10, color='red')

            # серые точки кадра
            lidar_ax.scatter(self.cart_lidar_frame[0, :], self.cart_lidar_frame[1, :], s=4, marker='o', color='gray')

            # # цветные точки после кластеризации
            # lidar_ax.scatter(cart_lidar_frame[0, :], cart_lidar_frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

            if vect_obstacles:
                for obst in vect_obstacles:
                    for i in range(obst.shape[1] - 1):
                        lidar_ax.plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
                                      linewidth=1.5, color='magenta')

            plt.draw()
            plt.pause(1 / self.fps)

            if plots:
                for plot in plots:
                    plot.remove()

            if time.time() - t0 > 0.1:
                assigned = True
                ax.clear()

            # time.sleep(1/fps)

    def execute_random_task(self):
        self.reset()

        # cmd_vel task
        for j in range(5):
            bot_lin_vel = uniform(0.5, 2)
            bot_ang_vel = uniform(-90, 90)
            self.execute_cmd_vel(bot_lin_vel, bot_ang_vel)
            if self.bot.terminated:
                break

        # self.p2p_motion(4)

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

fig, ax = plt.subplots()
ax.set_aspect('equal')

lidar_fig, lidar_ax = plt.subplots()
lidar_ax.set_aspect('equal')

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
bot_img = plt.Circle((x0, y0), process.bot.radius, color='r')
bot_nose = plt.Rectangle((x0 + 0.01 * sin(radians(dir0)),
                          y0 - 0.01 * sin(radians(dir0))),
                         process.bot.radius, 0.02,
                         angle=dir0, rotation_point='xy', color='black')

for ep in range(50):
    process.execute_random_task()


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
