import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from math import sin, cos, radians, pi, atan2, tan, sqrt, isinf, inf
from random import randint, uniform, normalvariate, random
import sklearn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import threading
import time
import csv

from Bot import Bot
from vectorization import getLines


def create_scene():
    contours = []
    room = np.asarray([[5, -5, -5, 5, 5], [5, 5, -5, -5, 5]], dtype=float)
    contours.append(room)

    obst_1 = np.asarray([[0.5, 2, 2, 0.5, 0.5], [2, 2, 4.49, 4.49, 2]], dtype=float)
    obst_2 = np.asarray([[-2, 0, 1, 0, -2, -2], [0, -1, -1, -2, -2, 0]], dtype=float)
    contours.append(obst_1)
    contours.append(obst_2)

    return contours

def get_lidar_frame(bot, contours, N, noise_std=0.1, lidar_angle=(pi - 0.001)):
    d_ang = lidar_angle / (N - 1)
    beam_angle_0 = (lidar_angle + pi) / 2
    cart_lidar_frame = np.zeros([3, N], dtype=float)
    cart_lidar_frame[2, :] = 1
    polar_lidar_frame = np.zeros([3, N], dtype=float)
    polar_lidar_frame[2, :] = 1
    dists = np.zeros([N], dtype=float)
    dists[:] = inf
    # seg_angles = np.zeros([2], dtype=float)

    B2W_T = np.asarray([[cos(radians(bot.dir - 90)), -sin(radians(bot.dir - 90)), bot.x],
                        [sin(radians(bot.dir - 90)), cos(radians(bot.dir - 90)), bot.y],
                        [0, 0, 1]], dtype=float)
    W2B_T = inv(B2W_T)

    for cnt in contours:
        seg_num = cnt.shape[1] - 1
        for i in range(seg_num):
            seg_st_W2B = W2B_T @ np.asarray([cnt[0, i], cnt[1, i], 1]).T
            seg_end_W2B = W2B_T @ np.asarray([cnt[0, i+1], cnt[1, i+1], 1]).T
            dx = seg_end_W2B[0] - seg_st_W2B[0]
            if dx == 0.0:
                seg_k = inf
                seg_b = seg_st_W2B[0]
            else:
                seg_k = (seg_end_W2B[1] - seg_st_W2B[1]) / (seg_end_W2B[0] - seg_st_W2B[0])
                seg_b = seg_st_W2B[1] - seg_k * seg_st_W2B[0]

            if seg_st_W2B[1] < 0.0 and seg_end_W2B[1] < 0.0:
                continue
            if seg_st_W2B[1] < 0.0:
                seg_st_W2B[1] = 0.0
                if not isinf(seg_k):
                    seg_st_W2B[0] = -seg_b / seg_k
            if seg_end_W2B[1] < 0.0:
                seg_end_W2B[1] = 0.0
                if not isinf(seg_k):
                    seg_end_W2B[0] = -seg_b / seg_k

            seg_angles = np.zeros([2], dtype=float)
            seg_angles[0] = atan2(seg_st_W2B[1], seg_st_W2B[0])
            seg_angles[1] = atan2(seg_end_W2B[1], seg_end_W2B[0])
            seg_angles.sort()

            for j in range(N):
                beam_angle = beam_angle_0 - d_ang * j
                beam_k = tan(beam_angle)
                beam_b = 0
                intsec_x, intsec_y = 0.0, 0.0
                if seg_angles[0] <= beam_angle <= seg_angles[1]:
                    if (seg_k == beam_k):
                        intsec_x, intsec_y = inf, inf  # прямые параллельны, в т.ч. и если обе вертикальные
                    else:
                        if isinf(beam_k):
                            intsec_x, intsec_y = beam_b, seg_k * beam_b + seg_b  # вертикальная исходная
                        elif isinf(seg_k):
                            intsec_x, intsec_y = seg_b, beam_k * seg_b + beam_b  # вертикальная подставляемая
                        else:
                            intsec_x = (beam_b - seg_b) / (seg_k - beam_k)
                            intsec_y = beam_k * (beam_b - seg_b) / (seg_k - beam_k) + beam_b
                    if intsec_y < 0:
                        continue
                    d = sqrt(intsec_x ** 2 + intsec_y ** 2)
                    if d < dists[j]:
                        dists[j] = d
                else:
                    continue
    for i in range(N):
        dist_noise = normalvariate(0.0, noise_std / 15)
        if isinf(dists[i]):
            cart_lidar_frame[0, i] = 0.0
            cart_lidar_frame[1, i] = 0.0
            polar_lidar_frame[0, i] = beam_angle_0 + i * d_ang
            polar_lidar_frame[1, i] = 0.0
        else:
            cart_lidar_frame[0, i] = (dists[i] + dist_noise) * cos(beam_angle_0 - d_ang * i)
            cart_lidar_frame[1, i] = (dists[i] + dist_noise) * sin(beam_angle_0 - d_ang * i)
            polar_lidar_frame[0, i] = beam_angle_0 + i * d_ang
            polar_lidar_frame[1, i] = dists[i]

    return cart_lidar_frame, polar_lidar_frame


def update_map(map, bot, lidar_frame, initial, prev_lidar_frame_B2W=None, threshold=0.05):
    B2W_T = np.asarray([[cos(radians(bot.dir - 90)), -sin(radians(bot.dir - 90)), bot.x],
                        [sin(radians(bot.dir - 90)), cos(radians(bot.dir - 90)), bot.y],
                        [0, 0, 1]], dtype=float)
    lidar_frame_B2W = B2W_T @ lidar_frame
    frame_to_append = []
    if not initial:
        frame_to_append = np.zeros([3, lidar_frame.shape[1]], dtype=float)
        count = 0
        for i in range(lidar_frame_B2W.shape[1]):
            j = 0
            while j < prev_lidar_frame_B2W.shape[1]:
                if sqrt((lidar_frame_B2W[0, i] - prev_lidar_frame_B2W[0, j]) ** 2 +
                        (lidar_frame_B2W[1, i] - prev_lidar_frame_B2W[1, j]) ** 2) < threshold:
                    break
                else:
                    j += 1
            if j == prev_lidar_frame_B2W.shape[1]:
                frame_to_append[:, count] = lidar_frame_B2W[:, i]
                count += 1
        if count == 0:
            return lidar_frame_B2W
    else:
        frame_to_append = lidar_frame_B2W
        count = lidar_frame_B2W.shape[1]

    map.append(frame_to_append[:, :count])
    # print(count)

    return lidar_frame_B2W


# def process_map(map):
    # TODO: мистический метод, в котором
    # 1. Проводится кластеризацию по контурам
    # 2. Точки в кластере сортируются по полярным углам ибо в текущей карте они неупорядочены
    # 3. Отсортированные контура векторизуются до ломаных
    # 4. Возвращаются контуры карты в виде ломаных
    # Это нужно для того, чтобы определять незнакомость объектов на лидаре не по точкам карты,
    # а по расстояниям до отрезков контуров на ней


def maintain_clustering(frame, eps=0.25, min_samples=2):
    # model = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=0.8)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    data = frame[:2, ].T
    model.fit(data)

    return model


def get_surrounding_objects(lidar_frame, clust_output):
    objects = []  # сохраняются в виде среза с кадра данных лидара
    clusters_inds = clust_output.labels_
    ind = 0
    fr = 0
    to = 1
    while ind < len(clusters_inds):
        if clusters_inds[ind] < 0:
            ind += 1
            continue
        else:
            fr = ind
            ind += 1
            while clusters_inds[ind] == clusters_inds[fr]:
                ind += 1
                if not ind < len(clusters_inds):
                    break
            to = ind
            objects.append(lidar_frame[:3, fr:to])
        if not ind < len(clusters_inds):
            break

    return objects


def detect_unfamiliar_objects(map, bot, objects, threshold=0.1):
    expansion = 0.025
    rois = []
    unfamiliar_objects = []
    B2W_T = np.asarray([[cos(radians(bot.dir - 90)), -sin(radians(bot.dir - 90)), bot.x],
                        [sin(radians(bot.dir - 90)), cos(radians(bot.dir - 90)), bot.y],
                        [0, 0, 1]], dtype=float)
    trans_map = map.T
    for object in objects:
        obj_B2W = B2W_T @ object
        xmin, ymin = np.min(obj_B2W[0, :]) - expansion, np.min(obj_B2W[1, :]) - expansion
        xmax, ymax = np.max(obj_B2W[0, :]) + expansion, np.max(obj_B2W[1, :]) + expansion
        ind1 = np.all(trans_map[:, :2] >= np.array([xmin, ymin]), axis=1)
        ind2 = np.all(trans_map[:, :2] <= np.array([xmax, ymax]), axis=1)

        map_roi = trans_map[np.logical_and(ind1, ind2)].T
        # rois.append(map_roi)

        if map_roi.size == 0:
            unfamiliar_objects.append(object)
        else:
            # print("OBJECT", object, object.shape)
            # print("ROI", map_roi, map_roi.shape)
            count = 0
            new_obj = np.zeros([3, obj_B2W.shape[1]], dtype=float)
            new_obj[2, :] = 1.0
            for i in range(obj_B2W.shape[1]):
                j = 0
                while j < map_roi.shape[1]:
                    if sqrt((obj_B2W[0, i] - map_roi[0, j]) ** 2 + (obj_B2W[1, i] - map_roi[1, j]) ** 2) < threshold:
                        break
                    else:
                        j += 1

                if j == map_roi.shape[1]:
                    new_obj[:2, count] = obj_B2W[:2, i]
                    count += 1
                elif count > 0:
                    unfamiliar_objects.append(inv(B2W_T) @ new_obj[:, :count])
                    count = 0
            if count > 0:
                unfamiliar_objects.append(inv(B2W_T) @ new_obj[:, :count])

            # if count > 0:
            #     unfamiliar_objects.append(new_obj[:, :count])

        # TODO: добавить выделение области на карте и проверка каждого объеката на наличие в этой области
        # print(len(unfamiliar_objects))
    return unfamiliar_objects
    # return rois


def save_map(map):
    with open('map.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for frame in map:
            frame_T = frame.T
            for pt in frame_T:
                writer.writerow(pt)


def read_map(file_name='map.csv'):
    map = np.genfromtxt(file_name, dtype=float, delimiter=' ')
    return map.T


def wait_for_bot_data(bot):
    while not bot.ready:
        time.sleep(0.001)
    bot.data_sent = True


def p2p_motion(x_goal, y_goal, dir_goal, lin_vel, fps, mapping=True):
    global prev_frame

    ax.clear()
    for cnt in contours:
        ax.plot(*cnt)

    bot_img = plt.Circle((bot.x, bot.y), bot.radius, color='r')
    bot_nose = plt.Rectangle((bot.x + 0.01 * sin(radians(bot.dir)),
                              bot.y - 0.01 * sin(radians(bot.dir))),
                             bot.radius, 0.02,
                             angle=bot.dir, rotation_point='xy', color='black')

    ax.add_patch(bot_img)
    ax.add_patch(bot_nose)

    motion = threading.Thread(target=bot.move_to_pnt, args=(x_goal, y_goal, dir_goal, lin_vel, fps))
    motion.start()

    while not bot.goal_reached:
        wait_for_bot_data(bot)
        bot_img.remove()
        bot_nose.remove()
        bot_img = plt.Circle((bot.x, bot.y), bot.radius, color='r')
        ax.add_patch(bot_img)
        bot_nose = plt.Rectangle((bot.x + 0.01 * sin(radians(bot.dir)),
                                  bot.y - 0.01 * sin(radians(bot.dir))),
                                 bot.radius, 0.02,
                                 angle=bot.dir, rotation_point='xy', color='black')
        ax.add_patch(bot_nose)

        lidar_ax.clear()
        frame, _ = get_lidar_frame(bot, contours, N, noise_std)
        lidar_ax.scatter(frame[0, :], frame[1, :], s=4, marker='o', color='gray')
        lidar_ax.scatter([0.0], [0.0], s=10, marker='o', color='red')

        if mapping:
            # add_frame_to_map(map, bot, frame)
            if not map:
                prev_frame = update_map(map, bot, frame, True)
            else:
                prev_frame = update_map(map, bot, frame, False, prev_lidar_frame_B2W=prev_frame, threshold=0.1)

        else:
            clustered = maintain_clustering(frame)
            objects = get_surrounding_objects(frame, clustered)
            obstacles = detect_unfamiliar_objects(map_from_file, bot, objects, threshold=0.1)

            for obst in obstacles:
                lidar_ax.scatter(obst[0, :], obst[1, :], s=10, marker='o', color='magenta')
            for obstacle in obstacles:
                lines = np.zeros([2, obstacle.shape[1]])
                Nlines = getLines(lines, obstacle, obstacle.shape[1], tolerance=0.1)
                for i in range(Nlines):
                    lidar_ax.plot([lines[0, i], lines[0, i + 1]], [lines[1, i], lines[1, i + 1]], linewidth=2)

        plt.draw()
        plt.pause(1/fps)

    # motion.join()


# TODO: добавить отдельный класс для лидара

# TODO: перестроить карту - чтобы было без пробелов или уже добить кластеризацию контуров и
#  работать уже с полигональной картой

map = []

N = 300
noise_std = 0.1
prev_frame = np.zeros([3, N], dtype=float)

bot = Bot()

fig, ax = plt.subplots()
ax.set_aspect('equal')

lidar_fig, lidar_ax = plt.subplots()
lidar_ax.set_aspect('equal')

contours = create_scene()
contours.append(np.asarray([[-4, -4, -3, -3, -4], [4, 3, 3, 4, 4]], dtype=float))
contours.append(np.asarray([[0, -0.5, -0.5, 0, 0], [4, 4, 3.5, 3.5, 4]], dtype=float))
contours.append(np.asarray([[-4, -4.5, -4.5, -4, -4], [0, 0, 0.5, 0.5, 0]], dtype=float))
contours.append(np.asarray([[4.4, 4.9, 4.9, 4.4, 4.4], [3, 3, 4, 4, 3]], dtype=float))
contours.append(np.asarray([[4.4, 4.9, 4.9, 4.4, 4.4], [-3, -3, -4, -4, -3]], dtype=float))
contours.append(np.asarray([[4, 3.5, 3.5, 4, 4], [-4, -4, -4.5, -4.5, -4]], dtype=float))
contours.append(np.asarray([[-1, 0, 0.5, -1], [-4.7, -4.3, -4.5, -4.7]], dtype=float))

map_from_file = read_map('map.csv')



p2p_motion(-2, 4, 0, 1, 10, mapping=False)
p2p_motion(-3, -4.2, 0, 1, 10, mapping=False)
p2p_motion(4, -3, 0, 1, 10, mapping=False)
p2p_motion(3.8, 0, 0, 1, 10, mapping=False)
p2p_motion(0, 0, 0, 1, 10, mapping=False)

# p2p_motion(-2, 4, 0, 1, 10)
# p2p_motion(-3, -4.2, 0, 1, 10)
# p2p_motion(4, -3, 0, 1, 10)
# p2p_motion(3.8, 0, 0, 1, 10)
# p2p_motion(0, 0, 0, 1, 10)




#
# save_map(map)
#


# bot.dir = 43
#
# for cnt in contours:
#     ax.plot(*cnt)
#
# bot_img = plt.Circle((bot.x, bot.y), bot.radius, color='r')
# bot_nose = plt.Rectangle((bot.x + 0.01 * sin(radians(bot.dir)),
#                           bot.y - 0.01 * sin(radians(bot.dir))),
#                          bot.radius, 0.02,
#                          angle=bot.dir, rotation_point='xy', color='black')
#
# ax.add_patch(bot_img)
# ax.add_patch(bot_nose)
#
# frame, _ = get_lidar_frame(bot, contours, N, noise_std)
#
# clustered = maintain_clustering(frame)
# objects = get_surrounding_objects(frame, clustered)
# obstacles = detect_unfamiliar_objects(map_from_file, bot, objects, threshold=0.05)
#
# for obstacle in obstacles:
#     lines = np.zeros([2, obstacle.shape[1]])
#     Nlines = getLines(lines, obstacle, obstacle.shape[1], tolerance=0.1)
#     for i in range(Nlines):
#         lidar_ax.plot([lines[0, i], lines[0, i + 1]], [lines[1, i], lines[1, i + 1]], linewidth=2)
#
# lidar_ax.scatter(frame[0, :], frame[1, :], s=4, marker='o', color='gray')
# for obst in obstacles:
#     lidar_ax.scatter(obst[0, :], obst[1, :], s=2, marker='o', color='magenta')
# # lidar_ax.scatter(frame[0, :], frame[1, :], s=4, c=clustered.labels_, cmap='rainbow')
# lidar_ax.scatter([0.0], [0.0], s=7, marker='o', color='red')



# map_from_file = read_map('map.csv')
# print(map_from_file.shape)
# # print(map_from_file.shape)



# model = maintain_clustering(map_from_file, eps=0.25)



# map_fig, map_ax = plt.subplots()
# map_ax.set_aspect('equal')
# # for fr in map:
# #     map_ax.scatter(fr[0, :], fr[1, :], s=1, marker='o', color='gray')
# map_ax.scatter(map_from_file[0, :], map_from_file[1, :], s=1, marker='o', color='gray')
# # for obst in obstacles:
# #     map_ax.scatter(obst[0, :], obst[1, :], s=5, marker='o', color='red')
# i = 2
# map_ax.scatter(obstacles[i][0, :], obstacles[i][1, :], s=5, marker='o', color='red')
# # map_ax.scatter(objects[0, :], objects[1, :], s=1, c=clustered.labels_, cmap='tab10')

plt.show()