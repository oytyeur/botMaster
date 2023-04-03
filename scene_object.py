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
    def __init__(self, nodes_coords, lin_vel=0.0, ang_vel=0.0, dir=0.0, movable=False):
        self.nodes_coords = np.zeros([3, nodes_coords.shape[1]], dtype=float)
        self.nodes_coords[:2, :] = nodes_coords
        self.nodes_coords[2, :] = 1.0
        self.movable = movable

        if movable:
            self.lin_vel = lin_vel
            self.ang_vel = ang_vel
            self.dir = dir

    # Движение объекта
    def transform(self, dt):
        self.nodes_coords[1, :] += self.lin_vel * dt
        if self.lin_vel > 0:
            if not np.min(self.nodes_coords[1, :]) < 2.5:
                self.nodes_coords[1, :] -= 5.75

        else:
            if not np.max(self.nodes_coords[1, :]) > -2.5:
                self.nodes_coords[1, :] += 5.75
