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


class SceneObject:
    def __init__(self, nodes_coords, lin_vel=0.0, ang_vel=0.0, dir=0.0, agent_radius=0.0, movable=False):
        self.nodes_coords = np.zeros([3, nodes_coords.shape[1]], dtype=float)
        self.nodes_coords[:2, :] = nodes_coords
        self.nodes_coords[2, :] = 1.0
        self.N = self.nodes_coords.shape[1] - 1

        # ТОЛЬКО ДЛЯ ПРАВИЛЬНЫХ МНОГОУГОЛЬНИКОВ
        self.c_x, self.c_y = np.mean(self.nodes_coords[0, :self.N]), np.mean(self.nodes_coords[1, :self.N])
        self.a = sqrt((self.nodes_coords[0, 0] - self.nodes_coords[0, 1]) ** 2 +
                      (self.nodes_coords[1, 0] - self.nodes_coords[1, 1]) ** 2)
        self.unsafe_radius = self.a / (2 * sin(pi / self.N)) + agent_radius + 0.05

        self.movable = movable

        if movable:
            self.lin_vel = lin_vel
            self.ang_vel = ang_vel
            self.dir = dir

    # Движение объекта
    def transform(self, dt):
        self.nodes_coords[1, :] += self.lin_vel * dt
        # self.c_x += self.lin_vel * dt
        self.c_y += self.lin_vel * dt
        if self.lin_vel > 0:
            if not np.min(self.nodes_coords[1, :]) < 3.0:
                self.nodes_coords[1, :] -= 7.5
                # self.c_x -= 5.75
                self.c_y -= 7.5
        else:
            if not np.max(self.nodes_coords[1, :]) > -3.0:
                self.nodes_coords[1, :] += 7.5
                # self.c_x += 5.75
                self.c_y += 7.5

    # Проверка на столкновение с объектом
    def check_agent_collision(self, bot):
        bot_c_x, bot_c_y, _ = bot.get_current_position()
        if sqrt((bot_c_x - self.c_x) ** 2 + (bot_c_y - self.c_y) ** 2) < self.unsafe_radius:
            bot.terminated = True
            # print('TERMINATED: Obstacle collision')
        return bot.terminated

