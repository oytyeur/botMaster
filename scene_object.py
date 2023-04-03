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
    def __init__(self, nodes_coords, movable=False):
        self.nodes_coords = np.zeros([3, nodes_coords.shape[1]], dtype=float)
        self.nodes_coords[:2, :] = nodes_coords
        self.nodes_coords[2, :] = 1.0
        self.movable = movable

        if movable:
            self.lin_vel = 0.0
            self.ang_vel = 0.0
            self.dir = 0.0

    # # Движение объекта
    # def move(self):
