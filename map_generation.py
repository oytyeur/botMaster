import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from math import sin, cos, radians, pi, atan2, tan, sqrt, isinf, inf
from random import randint, uniform, normalvariate, random
import sklearn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import threading

# cart_lidar_frame = np.zeros([2, N], dtype=float)
# polar_lidar_frame = np.zeros([2, N], dtype=float)
# # dists = np.zeros([N], dtype=float)
# pntsBuf = np.zeros([3, N], dtype=float)
# pntsPhi = np.zeros([N], dtype=float)

class Bot:
    def __init__(self):
        self.radius = 0.2
        self.x = 0.0
        self.y = 0.0
        self.dir = 0.0
        self.lin_vel = 0.0
        self.ang_vel = 0.0

def create_scene():
    contours = []
    room = np.asarray([[5, -5, -5, 5, 5], [5, 5, -5, -5, 5]], dtype=float)
    contours.append(room)

    obst_1 = np.asarray([[0.5, 2, 2, 0.5, 0.5], [2, 2, 4.8, 4.8, 2]], dtype=float)
    obst_2 = np.asarray([[-2, 0, 1, 0, -2, -2], [0, -1, -1, -2, -2, 0]], dtype=float)
    contours.append(obst_1)
    contours.append(obst_2)

    return contours

def get_lidar_frame(bot, contours, N, noise_std=0.1, lidar_angle=(pi - 0.001)):
    d_ang = lidar_angle / (N - 1)
    beam_angle_0 = (lidar_angle + pi) / 2
    cart_lidar_frame = np.zeros([3, N], dtype=float)
    cart_lidar_frame[2, :] = 1
    # polar_lidar_frame[:,:] = 0.0
    dists = np.zeros([N], dtype=float)
    dists[:] = inf
    # seg_angles = np.zeros([2], dtype=float)

    B2W_T = np.asarray([[cos(radians(bot.dir - 90)), -sin(radians(bot.dir - 90)), bot.x],
                        [sin(radians(bot.dir - 90)), cos(radians(bot.dir - 90)), bot.y],
                        [0, 0, 1]], dtype=float)
    # W2B_T = np.asarray([[cos(radians(bot.dir - 90)), sin(radians(bot.dir - 90)), -bot.y],
    #                     [-sin(radians(bot.dir- 90)), cos(radians(bot.dir - 90)), -bot.x],
    #                     [0, 0, 1]], dtype=float)
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
            # if seg_angles[0] < 0 and seg_angles[1] < 0:
            #     continue
            seg_angles.sort()
            # for c, ang in enumerate(seg_angles):
            #     if ang < 0:
            # if seg_angles[0] < 0:
            #     if seg_angles[1] < pi/2:
            #         seg_angles[0] = (lidar_angle - pi) / 2
            #     elif seg_angles[1] > pi/2:
            #         seg_angles[0] = (lidar_angle + pi) / 2
            #     else:
            #         if seg_angles[0] < -pi/2:
            #             seg_angles[0] = (lidar_angle + pi) / 2
            #         elif seg_angles[0] > -pi/2:
            #             seg_angles[0] = (lidar_angle - pi) / 2
            #         else:
            #             raise Exception()
            #     seg_angles.sort()

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
        dist_noise = normalvariate(0.0, noise_std / 3)
        if isinf(dists[i]):
            cart_lidar_frame[0, i] = 0.0
            cart_lidar_frame[1, i] = 0.0
        else: 
            cart_lidar_frame[0, i] = (dists[i] + dist_noise) * cos(beam_angle_0 - d_ang * i)
            cart_lidar_frame[1, i] = (dists[i] + dist_noise) * sin(beam_angle_0 - d_ang * i)

    return cart_lidar_frame

def maintain_clustering(lidar_frame):
    # model = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold=0.8)
    model = DBSCAN(eps=0.5, min_samples=3)
    data = lidar_frame[:2, ].T
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
            objects.append(lidar_frame[:2, fr:to])
        if not ind < len(clusters_inds):
            break

    return objects


N = 200
noise_std = 0.1

bot = Bot()

fig, ax = plt.subplots()
ax.set_aspect('equal')

lidar_fig, lidar_ax = plt.subplots()
lidar_ax.set_aspect('equal')
lidar_ax.set(xlim=(-10, 10), ylim=(-10, 10))

obj_fig, obj_ax = plt.subplots()
obj_ax.set_aspect('equal')

contours = create_scene()

# bot.x = 0
# bot.y = -4
# bot.dir = 15.0
# bot.lin_vel = 0.5

bot.x = 0
bot.y = 0
bot.dir = 135

n_episodes = 1000
dt = 0.5

for cnt in contours:
    ax.plot(*cnt)

for p in range(n_episodes):
    # bot_img = plt.Circle((bot.x, bot.y), bot.radius, color='r')
    # ax.add_patch(bot_img)
    # bot_nose = plt.Rectangle((bot.x + 0.01 * sin(radians(bot.dir)),
    #                           bot.y - 0.01 * sin(radians(bot.dir))),
    #                          bot.radius, 0.02,
    #                          angle=bot.dir, rotation_point='xy', color='black')
    # ax.add_patch(bot_nose)

    lidar_ax.clear()
    obj_ax.clear()
    frame = get_lidar_frame(bot, contours, N, noise_std)
    clustered = maintain_clustering(frame)
    obj = get_surrounding_objects(frame, clustered)
    for o in obj:
        obj_ax.scatter(o[0, :], o[1, :], s=5, marker='o', color='gray')
    # print(obj)
    # print(clustered.labels_)
    # print(len(set(clustered.labels_)))
    # lidar_ax.scatter(frame[0, :], frame[1, :], s=5, marker='o', color='gray')
    lidar_ax.scatter(frame[0, :], frame[1, :], s=5, c=clustered.labels_, cmap='tab10')
    lidar_ax.scatter([0.0], [0.0], s=7, marker='o', color='red')

    plt.draw()
    plt.pause(dt)
    # bot_img.remove()
    # bot_nose.remove()

    bot.x += bot.lin_vel * dt * cos(radians(bot.dir))
    bot.y += bot.lin_vel * dt * sin(radians(bot.dir))
    # bot.dir += 1.2
    bot.dir += 3.5

plt.show()