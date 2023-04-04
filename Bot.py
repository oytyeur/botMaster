from math import atan2, sin, cos, degrees, pi, sqrt, radians
import time
import threading

import numpy as np


class Bot:
    def __init__(self, discr_dt):
        self.radius = 0.2
        self.x = 0.0
        self.y = 0.0
        self.dir = 0.0

        self.lin_vel = 0.0
        self.ang_vel = 0.0

        self.DISCR_dT = discr_dt

        self.aligned = False

        self.ang_step = False
        self.lin_step = False

        self.motion_allowed = False
        self.goal_reached = False

        self.terminated = False

        self.lock = threading.Lock()
        self.motion_thread = threading.Thread(target=self.move, daemon=True)

        self.start()

    def start(self):
        self.motion_allowed = True
        self.cmd_vel(0.0, 0.0)
        self.motion_thread.start()

    # остановить робота вообще
    def stop(self):
        self.motion_allowed = False
        self.cmd_vel(0.0, 0.0)

    # TODO: РАЗОБРАТЬСЯ
    # Доля движения с заданной скоростью
    def move_dt(self):
        self.x += self.lin_vel * self.DISCR_dT * cos(radians(self.dir))
        self.y += self.lin_vel * self.DISCR_dT * sin(radians(self.dir))
        if not self.terminated:
            if self.check_map_collision():
                self.cmd_vel(0.0, 0.0)
        self.dir += (self.ang_vel * self.DISCR_dT) % 360.0
        if self.dir > 180:
            self.dir -= 360
        elif not self.dir > -180:
            self.dir += 360

    # процесс движения
    def move(self):
        while True:
            time.sleep(self.DISCR_dT)
            if self.motion_allowed:  # Плевать он хотел на это условие
                self.move_dt()

    # Задание команды скоростей
    def cmd_vel(self, lin_vel, ang_vel):
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel

    def get_current_position(self):
        return self.x, self.y, self.dir

    # поместить в определённую позицию
    def set_position(self, x, y, dir=0.0):
        self.x = x
        self.y = y
        self.dir = dir

    # Движение в точку, проверка положения
    # TODO: обавить развороты и против часовой стрелки
    def move_to_pnt_check(self, x_g, y_g, dir_g, lin_vel, fps):
        dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
        if dist > 0:
            self.goal_reached = False
            path_dir = degrees(atan2(y_g - self.y, x_g - self.x))
            if not self.aligned:
                if self.ang_step:
                    self.dir = path_dir
                    self.ang_step = False
                    self.aligned = True
                    self.motion_allowed = True

                else:
                    if abs(path_dir - self.dir) > 0 and not self.aligned:
                        # if self.dir > 0 and path_dir < 0:
                        # if path_dir < 0:
                        #     path_dir += 360
                        self.cmd_vel(0.0, 144.0 * np.sign(path_dir - self.dir))
                        if not abs(path_dir - self.dir) > abs(self.ang_vel * (1/fps)):
                            self.motion_allowed = False
                            self.cmd_vel(0.0, 0.0)
                            # if path_dir > 180:
                            #     path_dir -= 360
                            self.ang_step = True
                    else:
                        self.aligned = True

            else:
                if self.lin_step:
                    self.x = x_g
                    self.y = y_g
                    self.goal_reached = True
                    self.lin_step = False
                    print("Goal reached")
                    self.aligned = False
                    self.motion_allowed = True
                else:
                    self.cmd_vel(lin_vel, 0)
                    if not dist > self.lin_vel * (1/fps):
                        self.motion_allowed = False
                        self.cmd_vel(0.0, 0.0)
                        self.lin_step = True

        else:
            self.goal_reached = True
            print("Goal reached")

        return self.get_current_position()

    # Проверка столкновения с закартированными объектами
    def check_map_collision(self):
        c_x, c_y, _ = self.get_current_position()
        if not (-0.75 < c_x < 9.75) or not (-2.25 < c_y < 2.25):
            self.terminated = True
            print('TERMINATED: Wall collision')
        return self.terminated

