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

from scene_object import SceneObject


class Environment:
    # создание сцены
    def __init__(self):
        self.objects = []
        # room_coords = np.asarray([[5, -5, -5, 5, 5], [5, 5, -5, -5, 5]], dtype=float)
        room_coords = np.asarray([[10, -1, -1, 10, 10], [2.5, 2.5, -2.5, -2.5, 2.5]], dtype=float)
        room = SceneObject(room_coords)
        self.objects.append(room)

        # obst_1 = np.asarray([[0.5, 2, 2, 0.5, 0.5], [2, 2, 4.4, 4.4, 2]], dtype=float)
        # obst_2 = np.asarray([[-2, 0, 1, 0, -2, -2], [0, -1, -1, -2, -2, 0]], dtype=float)
        # contours.append(obst_1)
        # contours.append(obst_2)

    # добавить объект
    def add_object(self, nodes_coords, lin_vel=0, ang_vel=0, dir=0.0, agent_radius=0.0, movable=False):
        obj = SceneObject(nodes_coords, lin_vel, ang_vel, dir, agent_radius, movable)
        self.objects.append(obj)

    # удалить объект
    def remove_object(self, obj):
        self.objects.remove(obj)


    # # обновление сцены
    # def update_scene(self):

# scene = Environment()
# new_object = np.asarray([[-0.25, 0.25, 0.25, -0.25, -0.25], [2.25, 2.25, 1.75, 1.75, 2.25]], dtype=float)
# scene.add_object(new_object, movable=True)
#
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
#
# for obj in scene.objects:
#     print(obj.nodes_coords)
#     ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :])
#
# plt.show()
