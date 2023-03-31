from math import atan2, sin, cos, degrees, pi, sqrt, radians
import time


class Bot:
    def __init__(self):
        self.radius = 0.2
        self.x = 0.0
        self.y = 0.0
        self.dir = 0.0
        self.lin_vel = 0.0
        self.ang_vel = 0.0

        self.ready = True
        self.data_sent = False

        self.motion_allowed = False

        self.goal_reached = False

    # Ожидание посылки данных о движении
    def wait_for_data_sent(self):
        while not self.data_sent:
            time.sleep(0.001)
        self.data_sent = False

    # TODO: производить долю движения постоянно при вызове команды cmd_vel даже с нулевой скоростью в отдельном потоке
    # поток имеет функцию start/stop

    def start(self, fps):
        self.motion_allowed = True
        self.cmd_vel(0.0, 0.0)
        while self.motion_allowed:
            self.move_dt(1/fps)

    # Доля движения с заданной скоростью
    def move_dt(self, dt):
        self.x += self.lin_vel * dt * cos(radians(self.dir))
        self.y += self.lin_vel * dt * sin(radians(self.dir))
        self.dir += self.ang_vel * dt % 360.0

    # Задание команды скоростей
    def cmd_vel(self, lin_vel, ang_vel):
        self.lin_vel = lin_vel
        self.ang_vel = ang_vel

    # Движение в точку
    def move_to_pnt(self, x_g, y_g, dir_g, lin_vel, fps):
        self.goal_reached = False
        dt = 1/fps
        path_dir = degrees(atan2(y_g - self.y, x_g - self.x))
        self.cmd_vel(0.0, 45.0)
        while not self.dir == path_dir:
            self.ready = False
            time.sleep(dt)
            if self.dir > 0 and path_dir < 0:
                path_dir += 360

            if not path_dir - self.dir > self.ang_vel * dt:
                if path_dir > 180:
                    path_dir -= 360
                self.dir = path_dir

            else:
                self.dir += self.ang_vel * dt % 360.0
            self.ready = True
            self.wait_for_data_sent()

        self.cmd_vel(lin_vel, 0)
        dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
        while dist > 0.0:
            self.ready = False
            time.sleep(dt)
            if dist < self.lin_vel * dt:
                self.x = x_g
                self.y = y_g
                dist = 0.0
            else:
                self.x += self.lin_vel * dt * cos(radians(self.dir))
                self.y += self.lin_vel * dt * sin(radians(self.dir))
                dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
            self.ready = True
            self.wait_for_data_sent()

        # while not self.dir == dir_g:
        #     time.sleep(dt)
        #     if (2*pi + dir_g) - (2*pi + self.dir) <= self.ang_vel * dt:
        #         self.dir = dir_g
        #     else:
        #         self.dir += self.ang_vel * dt

        self.goal_reached = True
        self.ready = True
        self.data_sent = False

        print("Goal reached")