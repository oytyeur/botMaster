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


# Вернуть индексы
def check_potential_collision(vect_obstacles):
    return 0


# Движение из точки в точку
def p2p_motion(x_goal, y_goal, dir_goal, lin_vel, scene, bot, fps, beams_num=100, wanna_watch=True):
    global bot_img, bot_nose
    # ax.clear()
    # for obj in scene.objects:
    #     ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :])
    #
    # c_x, c_y, c_dir = bot.get_current_position()
    #
    # bot_img = plt.Circle((c_x, c_y), bot.radius, color='r')
    # bot_nose = plt.Rectangle((c_x + 0.01 * sin(radians(c_dir)),
    #                           c_y - 0.01 * sin(radians(c_dir))),
    #                          bot.radius, 0.02,
    #                          angle=c_dir, rotation_point='xy', color='black')
    #
    # ax.add_patch(bot_img)
    # ax.add_patch(bot_nose)
    # plots = []

    ax.scatter([x_goal], [y_goal], marker='*', s=100)

    while not bot.goal_reached:
        c_x, c_y, c_dir = bot.move_to_pnt_check(x_goal, y_goal, dir_goal, lin_vel, fps)
        # print(c_x, c_y, c_dir)

        # ax.clear()
        # for obj in scene.objects:
        #     ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :])

        # if plots:
        #     for plot in plots:
        #         plot.remove()

        plots = []
        for obj in scene.objects:
            plot, = ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='grey')
            if obj.movable:
                plots.append(plot)

        for obj in scene.objects:
            if obj.movable:
                obj.transform(1/fps)
        # scene.objects[2].transform(1/fps)

        bot_img = plt.Circle((c_x, c_y), bot.radius, color='r')
        bot_nose = plt.Rectangle((c_x + 0.01 * sin(radians(c_dir)),
                                  c_y - 0.01 * sin(radians(c_dir))),
                                 bot.radius, 0.02,
                                 angle=c_dir, rotation_point='xy', color='black')

        ax.add_patch(bot_img)
        ax.add_patch(bot_nose)

        # bot_img.remove()
        # bot_nose.remove()
        # bot_img = plt.Circle((c_x, c_y), bot.radius, color='r')
        # ax.add_patch(bot_img)
        # bot_nose = plt.Rectangle((c_x + 0.01 * sin(radians(c_dir)),
        #                           c_y - 0.01 * sin(radians(c_dir))),
        #                          bot.radius, 0.02,
        #                          angle=c_dir, rotation_point='xy', color='black')
        # ax.add_patch(bot_nose)

        lidar_ax.clear()

        frame, _ = get_lidar_frame(c_x, c_y, c_dir, scene.objects, beams_num)

        # лучи лидара
        # for i in range(frame.shape[1]):
        #     lidar_ax.plot([0.0, frame[0, i]], [0.0, frame[1, i]],
        #                   linewidth=0.1, color='red')

        clustered = maintain_frame_clustering(frame, eps=0.4)
        objects = get_surrounding_objects(frame, clustered)
        obstacles = detect_unfamiliar_objects(map_from_file, c_x, c_y, c_dir, objects, threshold=0.1)
        vect_obstacles = get_vectorized_obstacles(obstacles)

        # ось лидара
        lidar_ax.scatter([0.0], [0.0], s=10, color='red')

        # серые точки кадра
        lidar_ax.scatter(frame[0, :], frame[1, :], s=4, marker='o', color='gray')

        # # цветные точки после кластеризации
        # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

        # for obst in obstacles:
        #     nodes = np.zeros([2, obst.shape[1]], dtype=float)
        #     nodes_idx = []
        #     lines_num = getLinesSaM(nodes, nodes_idx, obst, tolerance=0.1)
        #     for i in range(lines_num):
        #         lidar_ax.plot([nodes[0, i], nodes[0, i + 1]], [nodes[1, i], nodes[1, i + 1]],
        #                          linewidth=1.5, color='magenta')

        if vect_obstacles:
            for obst in vect_obstacles:
                for i in range(obst.shape[1]-1):
                    lidar_ax.plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
                                  linewidth=1.5, color='magenta')

        # if wanna_watch:
        #     vis.visualize(scene, bot, c_x, c_y, c_dir, frame, vect_obstacles)

        plt.draw()
        plt.pause(1/fps)

        if plots:
            for plot in plots:
                plot.remove()

        bot_img.remove()
        bot_nose.remove()

    bot.goal_reached = False


def wait_for_assignment(bot, scene, wanna_watch=True):
    c_x, c_y, c_dir = bot.get_current_position()

    print('Waiting for assignment')

    ax.add_patch(bot_img)
    ax.add_patch(bot_nose)

    assigned = False
    t0 = time.time()

    plots = []

    while not assigned:
        plots = []
        for obj in scene.objects:
            plot, = ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :], color='grey')
            if obj.movable:
                plots.append(plot)

        for obj in scene.objects:
            if obj.movable:
                obj.transform(1 / fps)

            # print(plot)
            # for i in range(obj.nodes_coords.shape[1]-1):
            #     line, = ax.plot(obj.nodes_coords[0, i], obj.nodes_coords[1, i])
            #     print(line)
                # if obj.movable:
                #     plots.append(line)

        lidar_ax.clear()
        frame, _ = get_lidar_frame(c_x, c_y, c_dir, scene.objects, beams_num)

        # лучи лидара
        # for i in range(frame.shape[1]):
        #     lidar_ax.plot([0.0, frame[0, i]], [0.0, frame[1, i]],
        #                   linewidth=0.1, color='red')

        clustered = maintain_frame_clustering(frame, eps=0.4)
        objects = get_surrounding_objects(frame, clustered)
        obstacles = detect_unfamiliar_objects(map_from_file, c_x, c_y, c_dir, objects, threshold=0.1)
        vect_obstacles = get_vectorized_obstacles(obstacles)

        # if wanna_watch:
        #     vis.visualize(scene, bot, c_x, c_y, c_dir, frame, vect_obstacles)


        # ось лидара
        lidar_ax.scatter([0.0], [0.0], s=10, color='red')

        # серые точки кадра
        lidar_ax.scatter(frame[0, :], frame[1, :], s=4, marker='o', color='gray')

        # # цветные точки после кластеризации
        # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

        if vect_obstacles:
            for obst in vect_obstacles:
                for i in range(obst.shape[1] - 1):
                    lidar_ax.plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
                                  linewidth=1.5, color='magenta')

        plt.draw()
        plt.pause(1 / fps)

        if plots:
            for plot in plots:
                plot.remove()

        if time.time() - t0 > 0.25:
            assigned = True
            ax.clear()

        # time.sleep(1/fps)


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

def get_random_task(bot):
    # global scene
    bot.set_position(uniform(-0.5, 0), uniform(-2.0, 2.0))
    x_goal = uniform(8.5, 9.5)
    y_goal = uniform(-2.0, 2.0)
    n_obj = int(uniform(5, 7))
    scene = Environment()
    x_offset = 2.0
    for i in range(n_obj):
        s = uniform(0.5, 0.75)
        x = x_offset + uniform(0.75, 1.5)
        y = uniform(-2.5, 2.5)
        lin_vel = uniform(-4.0, 4.0)
        new_obj_nodes = np.asarray([[x - s/2, x + s/2, x + s/2, x - s/2, x - s/2],
                                    [y + s/2, y + s/2, y - s/2, y - s/2, y + s/2]], dtype=float)
        scene.add_object(new_obj_nodes, lin_vel=lin_vel, movable=True)
        x_offset = x

    p2p_motion(x_goal, y_goal, 0, 4, scene, bot, fps, beams_num=300)
    wait_for_assignment(bot, scene)
    # for obj in scene.objects:
    #     if obj.movable:
    #         scene.objects.remove(obj)


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

# ax.add_patch(bot_img)
# ax.add_patch(bot_nose)


# Движение по карте с незнакомыми препятствиями
map_from_file = read_map('map.csv')
beams_num = 300
# motion_lin_vel = 1
# sim_lin_vel = 2 * motion_lin_vel

# TODO: при возникновении тупняков на визуализации подобрать параметры
fps = 20  # ЭТО СТРОГО
discr_dt = 0.01  # 0.01 - 0.1 норм?

bot = Bot(discr_dt)


c_x, c_y, c_dir = bot.get_current_position()
bot_img = plt.Circle((c_x, c_y), bot.radius, color='r')
bot_nose = plt.Rectangle((c_x + 0.01 * sin(radians(c_dir)),
                          c_y - 0.01 * sin(radians(c_dir))),
                         bot.radius, 0.02,
                         angle=c_dir, rotation_point='xy', color='black')

for i in range(50):
    get_random_task(bot)


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
