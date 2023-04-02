from math import atan2, sin, cos, degrees, pi, sqrt, radians
import time
import threading
from multiprocessing import Process


class Bot:
    def __init__(self, discr_dt):
        self.radius = 0.2
        self.x = 0.0
        self.y = 0.0
        self.dir = 0.0

        self.temp_x = 0.0
        self.temp_y = 0.0
        self.temp_dir = 0.0

        self.lin_vel = 0.0
        self.ang_vel = 0.0

        self.DISCR_dT = discr_dt

        self.ready = False  # готовы ли данные о положении к считыванию
        self.data_sent = False  # посланы ли (считаны) данные о положении

        self.dir_g = 0.0
        self.aligned = False

        self.ang_step = False
        self.lin_step = False

        self.motion_allowed = False
        self.goal_reached = False

        self.lock = threading.Lock()
        self.motion_thread = threading.Thread(target=self.move)
        # self.motion_thread = Process(target=self.move, daemon=True)

        self.start()

    # Ожидание посылки данных о движении
    def wait_for_data_sent(self):
        while not self.data_sent:
            time.sleep(self.DISCR_dT)
        self.data_sent = False

    # TODO: производить долю движения постоянно при вызове команды cmd_vel даже с нулевой скоростью в отдельном потоке
    # поток имеет функцию start/stop

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
        self.temp_x += self.lin_vel * self.DISCR_dT * cos(radians(self.temp_dir))
        self.temp_y += self.lin_vel * self.DISCR_dT * sin(radians(self.temp_dir))
        self.temp_dir += (self.ang_vel * self.DISCR_dT) % 360.0
        if self.ready:
            self.x = self.temp_x
            self.y = self.temp_y
            self.dir = self.temp_dir
            self.ready = False

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

    # # Движение в точку
    # def move_to_pnt(self, x_g, y_g, dir_g, lin_vel, fps):
    #     self.goal_reached = False
    #     dt = 1/fps
    #     path_dir = degrees(atan2(y_g - self.y, x_g - self.x))
    #     self.cmd_vel(0.0, 45.0)
    #     while not self.dir == path_dir:
    #         self.ready = False
    #         time.sleep(dt)
    #         if self.dir > 0 and path_dir < 0:
    #             path_dir += 360
    #
    #         if not path_dir - self.dir > self.ang_vel * dt:
    #             if path_dir > 180:
    #                 path_dir -= 360
    #             self.dir = path_dir
    #
    #         else:
    #             self.dir += self.ang_vel * dt % 360.0
    #         self.ready = True
    #         self.wait_for_data_sent()
    #
    #     self.cmd_vel(lin_vel, 0)
    #     dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
    #     while dist > 0.0:
    #         self.ready = False
    #         time.sleep(dt)
    #         if dist < self.lin_vel * dt:
    #             self.x = x_g
    #             self.y = y_g
    #             dist = 0.0
    #         else:
    #             self.x += self.lin_vel * dt * cos(radians(self.dir))
    #             self.y += self.lin_vel * dt * sin(radians(self.dir))
    #             dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
    #         self.ready = True
    #         self.wait_for_data_sent()
    #
    #     # while not self.dir == dir_g:
    #     #     time.sleep(dt)
    #     #     if (2*pi + dir_g) - (2*pi + self.dir) <= self.ang_vel * dt:
    #     #         self.dir = dir_g
    #     #     else:
    #     #         self.dir += self.ang_vel * dt
    #
    #     self.goal_reached = True
    #     self.ready = True
    #     self.data_sent = False
    #
    #     print("Goal reached")



    # # Движение в точку
    # def move_to_pnt(self, x_g, y_g, dir_g, lin_vel):
    #     self.goal_reached = False
    #     dt = self.DISCR_dT
    #     path_dir = degrees(atan2(y_g - self.y, x_g - self.x))
    #     self.cmd_vel(0.0, 72.0)
    #     while not self.dir == path_dir:
    #         # self.ready = False
    #         # time.sleep(dt)
    #         if self.dir > 0 and path_dir < 0:
    #             path_dir += 360
    #
    #         if not path_dir - self.dir > self.ang_vel * dt:
    #             if path_dir > 180:
    #                 path_dir -= 360
    #             self.dir = path_dir
    #             self.cmd_vel(0.0, 0.0)
    #
    #         # else:
    #         #     self.dir += self.ang_vel * dt % 360.0
    #         # self.ready = True
    #         # self.wait_for_data_sent()
    #
    #     self.cmd_vel(lin_vel, 0)
    #     dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
    #     while dist > 0.0:
    #         # self.ready = False
    #         # time.sleep(dt)
    #         if not dist > self.lin_vel * dt:
    #             self.x = x_g
    #             self.y = y_g
    #             dist = 0.0
    #             self.cmd_vel(0.0, 0.0)
    #         else:
    #             # self.x += self.lin_vel * dt * cos(radians(self.dir))
    #             # self.y += self.lin_vel * dt * sin(radians(self.dir))
    #             dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
    #         # self.ready = True
    #         # self.wait_for_data_sent()
    #
    #     # while not self.dir == dir_g:
    #     #     time.sleep(dt)
    #     #     if (2*pi + dir_g) - (2*pi + self.dir) <= self.ang_vel * dt:
    #     #         self.dir = dir_g
    #     #     else:
    #     #         self.dir += self.ang_vel * dt
    #
    #     self.goal_reached = True
    #     # self.ready = True
    #     # self.data_sent = False
    #
    #     print("Goal reached")



    # # Движение в точку
    # def move_to_pnt_check(self, x_g, y_g, dir_g, lin_vel, fps):
    #     self.ready = True
    #     self.goal_reached = False
    #     dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
    #     if dist > 0:
    #         path_dir = degrees(atan2(y_g - self.y, x_g - self.x))
    #         # if not self.dir == path_dir:
    #         if abs(self.dir - path_dir) > 0:
    #             print(abs(self.dir - path_dir))
    #             self.cmd_vel(0.0, 72.0)
    #             # print("TURNING")
    #             if self.dir > 0 and path_dir < 0:
    #                 path_dir += 360
    #
    #             if not path_dir - self.dir > self.ang_vel * (1/fps):
    #                 if path_dir > 180:
    #                     path_dir -= 360
    #                 self.dir = path_dir
    #                 # self.cmd_vel(0.0, 0.0)
    #                 print("TURNING DONE")
    #
    #         else:
    #             self.cmd_vel(lin_vel, 0)
    #             print("GOING")
    #             # dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
    #             # if dist > 0.0:
    #             if not dist > self.lin_vel * (1/fps):
    #                 self.x = x_g
    #                 self.y = y_g
    #                 self.cmd_vel(0.0, 0.0)
    #                 self.goal_reached = True
    #                 print("Goal reached")
    #                 # else:
    #                 #     dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
    #
    #     else:
    #         self.goal_reached = True
    #         print("Goal reached")


# Движение в точку
    def move_to_pnt_check(self, x_g, y_g, dir_g, lin_vel, fps):
        self.ready = True
        dist = sqrt((x_g - self.x) ** 2 + (y_g - self.y) ** 2)
        if dist > 0:
            self.goal_reached = False
            path_dir = degrees(atan2(y_g - self.y, x_g - self.x))
            if not self.aligned:
                if self.ang_step:
                    self.temp_dir = path_dir
                    self.dir = path_dir
                    self.ang_step = False
                    self.aligned = True
                    self.motion_allowed = True

                else:
                    if abs(self.dir - path_dir) > 0 and not self.aligned:
                        self.cmd_vel(0.0, 72.0)
                        if self.dir > 0 and path_dir < 0:
                            path_dir += 360
                        if not path_dir - self.dir > self.ang_vel * (1/fps):
                            self.motion_allowed = False
                            self.cmd_vel(0.0, 0.0)
                            if path_dir > 180:
                                path_dir -= 360
                            self.ang_step = True

            else:
                if self.lin_step:
                    self.temp_x = x_g
                    self.temp_y = y_g
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