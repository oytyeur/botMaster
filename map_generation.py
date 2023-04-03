import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from numpy.linalg import inv
from math import sin, cos, radians, pi, atan2, tan, sqrt, isinf, inf
from random import randint, uniform, normalvariate, random
import sklearn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.spatial import ConvexHull
import threading
import multiprocessing
import time
import csv

from Bot import Bot
from vectorization import getLines, getLinesSaM
from lidar_processing import get_lidar_frame, maintain_frame_clustering, get_surrounding_objects, detect_unfamiliar_objects
from environment import Environment

# TODO аспределение по файлам-классам:
# Класс сцены: создание, редактирование, движение объектов
# Класс робота: движение, планирование пути
# Класс лидара: получение данных, обработка, детекция препятствий (векторизацию сюда же)
# Класс визуализатора: отрисовка, окна и пр.

# matplotlib.use('TkAgg')

# # Построение сцены
# def create_scene():
#     contours = []
#     room = np.asarray([[5, -5, -5, 5, 5], [5, 5, -5, -5, 5]], dtype=float)
#     contours.append(room)
#
#     obst_1 = np.asarray([[0.5, 2, 2, 0.5, 0.5], [2, 2, 4.4, 4.4, 2]], dtype=float)
#     obst_2 = np.asarray([[-2, 0, 1, 0, -2, -2], [0, -1, -1, -2, -2, 0]], dtype=float)
#     # contours.append(obst_1)
#     # contours.append(obst_2)
#
#     return contours


# TODO обрезать дальность обзора до метров 5, например
# Дополнение карты новым кадром
def update_map(map, c_x, c_y, c_dir, lidar_frame, initial, threshold=0.05):
    expansion = 0.05
    B2W_T = np.asarray([[cos(radians(c_dir - 90)), -sin(radians(c_dir - 90)), c_x],
                        [sin(radians(c_dir - 90)), cos(radians(c_dir - 90)), c_y],
                        [0, 0, 1]], dtype=float)
    lidar_frame_B2W = B2W_T @ lidar_frame
    if not initial:
        trans_map = map.T
        xmin, ymin = np.min(lidar_frame_B2W[0, :]) - expansion, np.min(lidar_frame_B2W[1, :]) - expansion
        xmax, ymax = np.max(lidar_frame_B2W[0, :]) + expansion, np.max(lidar_frame_B2W[1, :]) + expansion
        ind1 = np.all(trans_map[:, :2] >= np.array([xmin, ymin]), axis=1)
        ind2 = np.all(trans_map[:, :2] <= np.array([xmax, ymax]), axis=1)
        map_roi = trans_map[np.logical_and(ind1, ind2)].T

        frame_to_append = np.zeros([3, lidar_frame.shape[1]], dtype=float)
        count = 0
        for i in range(lidar_frame_B2W.shape[1]):
            j = 0
            while j < map_roi.shape[1]:
                if sqrt((lidar_frame_B2W[0, i] - map_roi[0, j]) ** 2 +
                        (lidar_frame_B2W[1, i] - map_roi[1, j]) ** 2) < threshold:
                    break
                else:
                    j += 1
            if j == map_roi.shape[1]:
                frame_to_append[:, count] = lidar_frame_B2W[:, i]
                count += 1
        if count == 0:
            return map
        else:
            map = np.append(map, frame_to_append[:, :count], axis=1)
            return map
    else:
        return lidar_frame_B2W

# Кластеризация карты
def maintain_map_clustering(map, eps=0.4, min_samples=2):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    data = map[:2, ].T
    model.fit(data)

    return model

# Разделение карты - сортировка по кластерам
def get_map_contours(map, clust_output):
    sorted_clusters = np.sort(clust_output.labels_)
    idx = np.argsort(clust_output.labels_)
    sorted_map = map[:, idx]
    contours = []
    ind = 0
    fr = 0
    to = 1
    while ind < len(sorted_clusters):
        if sorted_clusters[ind] < 0:
            ind += 1
            continue
        else:
            fr = ind
            ind += 1
            while sorted_clusters[ind] == sorted_clusters[fr]:
                ind += 1
                if not ind < len(sorted_clusters):
                    break
            to = ind
            contours.append(sorted_map[:3, fr:to])
        if not ind < len(sorted_clusters):
            break

    return contours


def vectorize_map(map):
    # TODO: мистический метод, в котором
    # 1. Проводится кластеризацию по контурам ГОТОВО
    # 2. Точки в кластере сортируются
    # 3. Отсортированные контура векторизуются до ломаных
    # 4. Возвращаются контуры карты в виде ломаных
    # Это нужно для того, чтобы определять незнакомость объектов на лидаре не по точкам карты,
    # а по расстояниям до отрезков контуров на ней, что в целом, может уменьшить количество вычислений

    #1
    map_clustered = maintain_map_clustering(map, eps=0.4)

    map_fig, map_ax = plt.subplots()
    map_ax.set_aspect('equal')

    # # map_ax.scatter(map_from_file[0, :], map_from_file[1, :], s=1, marker='o', color='gray')
    map_ax.scatter(map[0, :], map[1, :], s=1, c=map_clustered.labels_, cmap='rainbow')

    #2
    map_contours = get_map_contours(map, map_clustered)
    sorted_map_contours = []
    for cnt in map_contours:
        print(cnt.shape)
        # dist_matrix = scipy.spatial.distance_matrix(cnt[:2, :].T, cnt[:2, :].T)
        # indices = np.arange(cnt.shape[1])
        # # sorted_cnt = np.zeros([cnt.shape[0], cnt.shape[1]], dtype=float)
        # sorted_cnt_ind = [np.random.randint(0, cnt.shape[1])]
        # while len(sorted_cnt_ind) < cnt.shape[1]:
        #     last_index = sorted_cnt_ind[-1]
        #     distances = dist_matrix[last_index, indices]
        #     next_index = indices[np.argmin(distances)]
        #
        #     sorted_cnt_ind.append(next_index)
        # sorted_cnt = cnt[:, sorted_cnt_ind]
        # sorted_map_contours.append(sorted_cnt)


        # for i in range(sorted_cnt.shape[1]):
        #     if i == sorted_cnt.shape[1] - 1:
        #         map_ax.plot([sorted_cnt[0, i], sorted_cnt[0, 0]], [sorted_cnt[1, i], sorted_cnt[1, 0]], linewidth=2)
        #     else:
        #         map_ax.plot([sorted_cnt[0, i], sorted_cnt[0, i + 1]], [sorted_cnt[1, i], sorted_cnt[1, i + 1]], linewidth=2)

    #3
    # for cnt in map_contours:
    #     lines = np.zeros([2, cnt.shape[1]])
    #     # lines[2, :] = 1.0
    #     Nlines = getLines(lines, cnt, cnt.shape[1], tolerance=0.1)
    #     for i in range(Nlines):
    #         map_ax.plot([lines[0, i], lines[0, i + 1]], [lines[1, i], lines[1, i + 1]], linewidth=2)

# Сохранение карты в csv
def save_map(map):
    with open('map.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for i in range(map.shape[1]):
            writer.writerow(map[:, i].T)

    print("Map saved to map.csv")

# Чтение карты из csv
def read_map(file_name='map.csv'):
    map = np.genfromtxt(file_name, dtype=float, delimiter=' ')
    return map.T

# Ожидание обновления движения
def wait_for_bot_data(bot):
    while not bot.ready:
        time.sleep(0.001)
    bot.data_sent = True

# TODO: вероятно, нужда в отдельном процессе есть только у лидара - ждать его кадра,
# движение робота можно осуществить, передав ему время вычислений текущей обстановки и прочего, то есть
# робот получил новое положение, в этом положении происходят получение кадра, обработка, принятие решения
# Хотя в процессе движения за время обработки данных он может проехать гораздо дальше положения, в котором шли вычисления
# Можно замерять время вычислений и после принятия решения о движении при


# Движение из точки в точку
def p2p_motion(x_goal, y_goal, dir_goal, lin_vel, scene, fps, beams_num=100, mapping=True, initial=False):
    global map
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
    # TODO: ПОЧЕМУ ТРОИТ ПРИ ВЫЗОВЕ В ОТДЕЛЬНОМ ПОТОКЕ
    # ptp_motion_thread = threading.Thread(target=bot.move_to_pnt, args=(x_goal, y_goal, dir_goal, lin_vel))
    # ptp_motion_thread.start()
    t0 = time.time()
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
        frame, _ = get_lidar_frame(c_x, c_y, c_dir, scene.objects, beams_num)

        # лучи лидара
        # for i in range(frame.shape[1]):
        #     lidar_ax.plot([0.0, frame[0, i]], [0.0, frame[1, i]],
        #                   linewidth=0.1, color='red')

        if initial:
            map = update_map(map, c_x, c_y, c_dir, frame, initial)
            initial = False
        else:
            map = update_map(map, c_x, c_y, c_dir, frame, initial, threshold=0.05)
        # серые точки кадра
        lidar_ax.scatter(frame[0, :], frame[1, :], s=4, marker='o', color='gray')
        # ось лидара
        lidar_ax.scatter([0.0], [0.0], s=10, color='red')

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


def generate_map(bot, scene, fps=10):
    # ПРЕДВАРИТЕЛЬНОЕ КАРТИРОВАНИЕ
    mapping_lin_vel = 2
    sim_lin_vel = 2 * mapping_lin_vel

    p2p_motion(-1.5, 4.5, 0, sim_lin_vel, scene, fps, initial=True)
    p2p_motion(-4, -4.2, 0, sim_lin_vel, scene, fps)
    p2p_motion(4, -3, 0, sim_lin_vel, scene, fps)
    p2p_motion(4.75, 4.75, 0, sim_lin_vel, scene, fps)
    p2p_motion(3, 0, 0, sim_lin_vel, scene, fps)
    p2p_motion(0, 0, 0, sim_lin_vel, scene, fps)
    save_map(map)



# # TODO: добить кластеризацию контуров и работать уже с полигональной картой
#
# # TODO: вынести константы в предвырительное объявление
# map = []
#
# scene = Environment()
# new_object = np.asarray([[-0.25, 0.25, 0.25, -0.25, -0.25], [2.25, 2.25, 1.75, 1.75, 2.25]], dtype=float)
# scene.add_object(new_object, movable=True)
#
# fps = 20
# discr_dt = 0.01
# bot = Bot(discr_dt)
#
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
#
# lidar_fig, lidar_ax = plt.subplots()
# lidar_ax.set_aspect('equal')
#
# # ПРОИЗВЕСТИ КАРТИРОВАНИЕ
# # contours = create_scene()
# generate_map(bot, scene, fps=fps)


# # # ПОКАЗАТЬ КАДР В НЕКОТОРОЙ ПОЗИЦИИ РОБОТА
# # # bot.x = 3
# # # bot.y = 1
# # # bot.dir = -90
# # # get_single_frame()
#
#
#
# # ПОКАЗАТЬ КАРТУ
# map_from_file = read_map('map.csv')
# vectorize_map(map_from_file)



# # map_clustered = maintain_map_clustering(map_from_file, eps=0.4)
# # # map_contours = get_map_contours(map_from_file, map_clustered)
# #
# # map_ax.scatter(map_from_file[0, :], map_from_file[1, :], s=1, marker='o', color='gray')
# # # map_ax.scatter(map_from_file[0, :], map_from_file[1, :], s=1, c=map_clustered.labels_, cmap='rainbow')
#
#
#
# # # map_contours = get_map_contours(map_from_file, map_clustered)
# # # for mp_cnt in map_contours:
# # #     nodes = np.zeros([2, mp_cnt.shape[1]])
# # #     Nlines = getLines(nodes, mp_cnt, mp_cnt.shape[1], tolerance=0.1)
# # #     for i in range(Nlines):
# # #         map_ax.plot([nodes[0, i], nodes[0, i + 1]], [nodes[1, i], nodes[1, i + 1]], linewidth=2)
#
# # # # # Это больше для работы с ROI нужно было
# # # # # for fr in map:
# # # # #     map_ax.scatter(fr[0, :], fr[1, :], s=1, marker='o', color='gray')
# # # # map_ax.scatter(map_from_file[0, :], map_from_file[1, :], s=1, marker='o', color='gray')
# # # # # for obst in obstacles:
# # # # #     map_ax.scatter(obst[0, :], obst[1, :], s=5, marker='o', color='red')
# # # # i = 2
# # # # map_ax.scatter(obstacles[i][0, :], obstacles[i][1, :], s=5, marker='o', color='red')
# # # # # map_ax.scatter(objects[0, :], objects[1, :], s=1, c=clustered.labels_, cmap='tab10')
#
# plt.show()