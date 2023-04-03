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


# Формирование кадра лидара
def get_lidar_frame(c_x, c_y, c_dir, objects, beams_num=100, noise_std=0.1, lidar_angle=(pi - 0.001)):
    d_ang = lidar_angle / (beams_num - 1)
    beam_angle_0 = (lidar_angle + pi) / 2
    cart_lidar_frame = np.zeros([3, beams_num], dtype=float)
    cart_lidar_frame[2, :] = 1
    polar_lidar_frame = np.zeros([3, beams_num], dtype=float)
    polar_lidar_frame[2, :] = 1
    dists = np.zeros([beams_num], dtype=float)
    dists[:] = inf

    B2W_T = np.asarray([[cos(radians(c_dir - 90)), -sin(radians(c_dir - 90)), c_x],
                        [sin(radians(c_dir - 90)), cos(radians(c_dir - 90)), c_y],
                        [0, 0, 1]], dtype=float)
    W2B_T = inv(B2W_T)

    for obj in objects:
        seg_num = obj.nodes_coords.shape[1] - 1
        for i in range(seg_num):
            seg_st_W2B = W2B_T @ np.asarray([obj.nodes_coords[0, i], obj.nodes_coords[1, i], 1]).T
            seg_end_W2B = W2B_T @ np.asarray([obj.nodes_coords[0, i+1], obj.nodes_coords[1, i+1], 1]).T
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

            for j in range(beams_num):
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
    for i in range(beams_num):
        dist_noise = normalvariate(0.0, noise_std / 10)
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


# Кластеризация кадра
def maintain_frame_clustering(frame, eps=0.4, min_samples=2):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    data = frame[:2, ].T
    model.fit(data)

    return model


# Выделение отдельных кластеров на кадре, исключение выбросов - сортировка по кластерам
def get_surrounding_objects(lidar_frame, clust_output):
    objects = []  # сохраняются в виде среза с кадра данных лидара
    sorted_clusters = clust_output.labels_
    ind = 0
    fr = 0
    to = 1
    while ind < len(sorted_clusters):
        if sorted_clusters[ind] < 0:
            ind += 1
            continue
        else:
            fr = ind
            if ind < len(sorted_clusters) - 1:
                ind += 1
            while sorted_clusters[ind] == sorted_clusters[fr]:
                ind += 1
                if not ind < len(sorted_clusters):
                    break
            to = ind
            objects.append(lidar_frame[:3, fr:to])
        if not ind < len(sorted_clusters):
            break

    return objects


# Обнаружение незнакомых препятствий с помощью карты
def detect_unfamiliar_objects(map, c_x, c_y, c_dir, objects, threshold=0.1):
    expansion = 0.05
    unfamiliar_objects = []
    B2W_T = np.asarray([[cos(radians(c_dir - 90)), -sin(radians(c_dir - 90)), c_x],
                        [sin(radians(c_dir - 90)), cos(radians(c_dir - 90)), c_y],
                        [0, 0, 1]], dtype=float)
    trans_map = map.T
    for object in objects:
        obj_B2W = B2W_T @ object
        xmin, ymin = np.min(obj_B2W[0, :]) - expansion, np.min(obj_B2W[1, :]) - expansion
        xmax, ymax = np.max(obj_B2W[0, :]) + expansion, np.max(obj_B2W[1, :]) + expansion
        ind1 = np.all(trans_map[:, :2] >= np.array([xmin, ymin]), axis=1)
        ind2 = np.all(trans_map[:, :2] <= np.array([xmax, ymax]), axis=1)

        map_roi = trans_map[np.logical_and(ind1, ind2)].T

        if map_roi.size == 0:
            unfamiliar_objects.append(object)
        else:
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

    return unfamiliar_objects


