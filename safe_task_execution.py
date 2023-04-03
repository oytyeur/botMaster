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
from vectorization import getLines, getLinesSaM
from map_generation import read_map
from environment import Environment
from lidar_processing import get_lidar_frame, maintain_frame_clustering, get_surrounding_objects, detect_unfamiliar_objects


# TODO аспределение по файлам-классам:
# Класс сцены: создание, редактирование, движение объектов
# Класс робота: движение, планирование пути
# Класс лидара: получение данных, обработка, детекция препятствий (векторизацию сюда же)
# Класс визуализатора: отрисовка, окна и пр.


# TODO: вероятно, нужда в отдельном процессе есть только у лидара - ждать его кадра,
# движение робота можно осуществить, передав ему время вычислений текущей обстановки и прочего, то есть
# робот получил новое положение, в этом положении происходят получение кадра, обработка, принятие решения
# Хотя в процессе движения за время обработки данных он может проехать гораздо дальше положения, в котором шли вычисления
# Можно замерять время вычислений и после принятия решения о движении при


# Движение из точки в точку
def p2p_motion(x_goal, y_goal, dir_goal, lin_vel, scene, fps, beams_num=100):
    ax.clear()
    for obj in scene.objects:
        ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :])

    c_x, c_y, c_dir = bot.get_current_position()

    bot_img = plt.Circle((c_x, c_y), bot.radius, color='r')
    bot_nose = plt.Rectangle((c_x + 0.01 * sin(radians(c_dir)),
                              c_y - 0.01 * sin(radians(c_dir))),
                             bot.radius, 0.02,
                             angle=c_dir, rotation_point='xy', color='black')

    ax.add_patch(bot_img)
    ax.add_patch(bot_nose)

    t = 0.0

    while not bot.goal_reached:
        c_x, c_y, c_dir = bot.move_to_pnt_check(x_goal, y_goal, dir_goal, lin_vel, fps)

        bot_img.remove()
        bot_nose.remove()
        bot_img = plt.Circle((c_x, c_y), bot.radius, color='r')
        ax.add_patch(bot_img)
        bot_nose = plt.Rectangle((c_x + 0.01 * sin(radians(c_dir)),
                                  c_y - 0.01 * sin(radians(c_dir))),
                                 bot.radius, 0.02,
                                 angle=c_dir, rotation_point='xy', color='black')
        ax.add_patch(bot_nose)

        lidar_ax.clear()
        # t0 = time.time()
        frame, _ = get_lidar_frame(c_x, c_y, c_dir, scene.objects, beams_num)

        # лучи лидара
        # for i in range(frame.shape[1]):
        #     lidar_ax.plot([0.0, frame[0, i]], [0.0, frame[1, i]],
        #                   linewidth=0.1, color='red')

        clustered = maintain_frame_clustering(frame, eps=0.4)
        objects = get_surrounding_objects(frame, clustered)
        obstacles = detect_unfamiliar_objects(map_from_file, c_x, c_y, c_dir, objects, threshold=0.1)

        # ось лидара
        lidar_ax.scatter([0.0], [0.0], s=10, color='red')

        # серые точки кадра
        lidar_ax.scatter(frame[0, :], frame[1, :], s=4, marker='o', color='gray')

        # # цветные точки после кластеризации
        # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

        for obst in obstacles:
            nodes = np.zeros([2, obst.shape[1]], dtype=float)
            nodes_idx = []
            lines_num = getLinesSaM(nodes, nodes_idx, obst, tolerance=0.1)
            for i in range(lines_num):
                lidar_ax.plot([nodes[0, i], nodes[0, i + 1]], [nodes[1, i], nodes[1, i + 1]],
                                 linewidth=1.5, color='magenta')


        plt.draw()
        plt.pause(1/fps)

    bot.goal_reached = False


# def get_single_frame():
#     map_from_file = read_map('map.csv')
#     ax.clear()
#     for cnt in contours:
#         ax.plot(*cnt)
#
#     bot_img = plt.Circle((bot.x, bot.y), bot.radius, color='r')
#     bot_nose = plt.Rectangle((bot.x + 0.01 * sin(radians(bot.dir)),
#                               bot.y - 0.01 * sin(radians(bot.dir))),
#                              bot.radius, 0.02,
#                              angle=bot.dir, rotation_point='xy', color='black')
#
#     ax.add_patch(bot_img)
#     ax.add_patch(bot_nose)
#
#     frame, _ = get_lidar_frame(bot, contours, N, noise_std)
#     lidar_ax.scatter(frame[0, :], frame[1, :], s=3, marker='o', color='gray')
#
#     clustered = maintain_frame_clustering(frame, eps=0.4)
#     objects = get_surrounding_objects(frame, clustered)
#     obstacles = detect_unfamiliar_objects(map_from_file, bot, objects, threshold=0.1)
#
#     lidar_ax.scatter([0.0], [0.0], s=10, marker='o', c='red')
#
#     # for i in range(frame.shape[1]):
#     #     lidar_ax[0].plot([0.0, frame[0, i]], [0.0, frame[1, i]],
#     #                   linewidth=0.1, color='red')
#
#
#     # nodes = np.zeros([2, frame.shape[1]], dtype=float)
#     # nodes_idx = []
#     # lines_num = getLinesSaM(nodes, nodes_idx, frame, tolerance=0.1)
#     # for i in range(lines_num):
#     #     lidar_ax[1].plot([nodes[0, i], nodes[0, i + 1]], [nodes[1, i], nodes[1, i + 1]], linewidth=2)
#
#     for obst in obstacles:
#         nodes = np.zeros([2, obst.shape[1]], dtype=float)
#         nodes_idx = []
#         lines_num = getLinesSaM(nodes, nodes_idx, obst, tolerance=0.1)
#         for i in range(lines_num):
#             lidar_ax.plot([nodes[0, i], nodes[0, i + 1]], [nodes[1, i], nodes[1, i + 1]],
#                              linewidth=1.5, color='magenta')


# # TODO: вынести константы в предварительное объявление
# fps = 20
# discr_dt = 0.01
# bot = Bot(discr_dt)
#
# # scene_contours = create_scene()
# scene = Environment()
# new_object = np.asarray([[-0.25, 0.25, 0.25, -0.25, -0.25], [2.25, 2.25, 1.75, 1.75, 2.25]], dtype=float)
# scene.add_object(new_object, movable=True)
#
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
#
# lidar_fig, lidar_ax = plt.subplots()
# lidar_ax.set_aspect('equal')
#
#
#
# # # ДВИЖЕНИЕ С ИМЕЮЩЕЙСЯ КАРТОЙ, ОТРАБОТКА ТРАЕКТОРИИ
# # # Добавление незакартированных препятствий
# # scene_contours.append(np.asarray([[-4, -4, -3, -3, -4], [4, 3, 3, 4, 4]], dtype=float))
# # scene_contours.append(np.asarray([[0, -0.5, -0.5, 0, 0], [4, 4, 3.5, 3.5, 4]], dtype=float))
# # scene_contours.append(np.asarray([[-4, -4.5, -4.5, -4, -4], [0, 0, 0.5, 0.5, 0]], dtype=float))
# # scene_contours.append(np.asarray([[4.4, 4.9, 4.9, 4.4, 4.4], [3, 3, 4, 4, 3]], dtype=float))
# # scene_contours.append(np.asarray([[4.4, 4.9, 4.9, 4.4, 4.4], [-3, -3, -4, -4, -3]], dtype=float))
# # scene_contours.append(np.asarray([[4, 3.5, 3.5, 4, 4], [-4, -4, -4.5, -4.5, -4]], dtype=float))
# # scene_contours.append(np.asarray([[-1, 0, 0.5, -1], [-4.7, -4.3, -4.5, -4.7]], dtype=float))
#
#
# # Движение по карте с незнакомыми препятствиями
# map_from_file = read_map('map.csv')
# beams_num = 300
# motion_lin_vel = 2
# sim_lin_vel = 2 * motion_lin_vel
#
# # p2p_motion(4, 0, 0, motion_lin_vel, fps, beams_num=beams_num, mapping=False)
#
# p2p_motion(-2, 4, 0, sim_lin_vel, scene, fps, beams_num=beams_num)
# p2p_motion(-3, -4.2, 0, sim_lin_vel, scene, fps, beams_num=beams_num)
# p2p_motion(4, -3, 0, sim_lin_vel, scene, fps, beams_num=beams_num)
# p2p_motion(3.8, 0, 0, sim_lin_vel, scene, fps, beams_num=beams_num)
# p2p_motion(0, 0, 0, sim_lin_vel, scene, fps, beams_num=beams_num)



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
