import matplotlib.pyplot as plt
from math import sin, cos, radians, pi, atan2, tan, sqrt, isinf, inf

class Visualizer:
    def __init__(self, see_scene=True, see_lidar=True):
        self.see_scene = see_scene
        self.see_lidar = see_lidar
        if self.see_lidar:
            self.lidar_fig, self.lidar_ax = plt.subplots()
            self.lidar_ax.set_aspect('equal')

        if self.see_scene:
            self.fig, self.ax = plt.subplots()
            self.ax.set_aspect('equal')

    def visualize(self, scene, bot, c_x, c_y, c_dir, frame, vect_obstacles):

        if self.see_scene:
            self.ax.clear()

            for obj in scene.objects:
                self.ax.plot(obj.nodes_coords[0, :], obj.nodes_coords[1, :])

            # c_x, c_y, c_dir = bot.get_current_position()
            bot_img = plt.Circle((c_x, c_y), bot.radius, color='r')
            bot_nose = plt.Rectangle((c_x + 0.01 * sin(radians(c_dir)),
                                      c_y - 0.01 * sin(radians(c_dir))),
                                     bot.radius, 0.02,
                                     angle=c_dir, rotation_point='xy', color='black')

            self.ax.add_patch(bot_img)
            self.ax.add_patch(bot_nose)

        if self.see_lidar:
            self.lidar_ax.clear()

            # ось лидара
            self.lidar_ax.scatter([0.0], [0.0], s=10, color='red')

            # серые точки кадра
            self.lidar_ax.scatter(frame[0, :], frame[1, :], s=4, marker='o', color='gray')

            # # цветные точки после кластеризации
            # lidar_ax.scatter(frame[0, :], frame[1, :], s=1, c=clustered.labels_, cmap='tab10')

            if vect_obstacles:
                for obst in vect_obstacles:
                    for i in range(obst.shape[1]):
                        self.lidar_ax.plot([obst[0, i], obst[0, i + 1]], [obst[1, i], obst[1, i + 1]],
                                           linewidth=1.5, color='magenta')

        if self.see_lidar or self.see_scene:
            plt.draw()
            plt.pause(0.001)
