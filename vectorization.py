import math as m
from math import inf, isinf
import numpy as np


# ====== РЕАЛИЗАЦИЯ ФУНКЦИЙ МНК С ИСПОЛЬЗОВАНИЕМ NUMPY =======

def calc_loss_func(seg_pnts: np.ndarray, A: float, C: float):
    loss_func_value = 0
    for i in range(seg_pnts.shape[1]):
        # Расстояние до текущей прямой
        dist = abs(seg_pnts[0, i] * A - seg_pnts[1, i] + C) / m.sqrt(A ** 2 + 1)
        loss_func_value += dist ** 2
    return loss_func_value


def calc_sums(seg_pnts: np.ndarray, sums: np.ndarray):
    sums[0] = np.sum(seg_pnts[0])  # сумма всех х координат
    sums[1] = np.sum(seg_pnts[1])  # сумма всех у координат
    sums[2] = np.dot(seg_pnts[0], seg_pnts[1].T)  # сумма произведений всех х и всех у координат соответственно
    sums[3] = np.dot(seg_pnts[0], seg_pnts[0].T)  # сумма квадратов всех координат х
    sums[4] = np.dot(seg_pnts[1], seg_pnts[1].T)  # сумма квадратов всех координат у

    return sums


def calc_sums_step_back(pnt: np.ndarray, sums: np.ndarray):
    x = pnt[0, -1]
    y = pnt[1, -1]

    sums[0] -= x
    sums[1] -= y
    sums[2] -= x * y
    sums[3] -= x ** 2
    sums[4] -= y ** 2
    return sums


def line_approx_lsm(pnts: np.ndarray, fr: int, to: int, sums: np.ndarray, back=False):
    pts_num = to - fr
    if not back:
        sums = calc_sums(pnts[:2, fr:to], sums)
    else:
        sums = calc_sums_step_back(pnts[:2, fr:to], sums)
        pts_num -= 1
        to -= 1

    x_sum = sums[0]
    y_sum = sums[1]
    xy_sum = sums[2]
    x_sq_sum = sums[3]
    y_sq_sum = sums[4]
    # Вычисление A для минимумов функции потерь
    phi = xy_sum - x_sum * y_sum / pts_num
    theta = (x_sq_sum - y_sq_sum) / phi + (y_sum ** 2 - x_sum ** 2) / (pts_num * phi)
    D = theta ** 2 + 4  # дискриминант
    A1 = (-theta + m.sqrt(D)) / 2
    A2 = (-theta - m.sqrt(D)) / 2
    # Вычисление С для минимумов функции потерь
    C1 = (y_sum - x_sum * A1) / pts_num
    C2 = (y_sum - x_sum * A2) / pts_num
    # Подстановка в функцию потерь, выявление лучшего
    lf1 = calc_loss_func(pnts[:2, fr:to], A1, C1)
    lf2 = calc_loss_func(pnts[:2, fr:to], A2, C2)
    # Выбор наименьшего значения функции потерь, возврат соответствующих ему параметров А и С
    if lf1 < lf2:
        return A1, C1, np.mean(((A1 * pnts[0, fr:to] - pnts[1, fr:to] + C1) ** 2)) / (A1 ** 2 + 1), sums
    else:
        return A2, C2, np.mean(((A2 * pnts[0, fr:to] - pnts[1, fr:to] + C2) ** 2)) / (A2 ** 2 + 1), sums


# =====================================================================


def calc_intersection(A1, C1, A2, C2):
    x = (C2 - C1) / (A1 - A2)
    y = A2 * (C2 - C1) / (A1 - A2) + C2
    return x, y


def calc_normal(pnt, A1):
    x0 = pnt[0]
    y0 = pnt[1]
    A2 = -1 / A1
    C2 = x0 / A1 + y0
    return A2, C2


def getLines(lines: np.ndarray, pnts: np.ndarray, Npnts: int, tolerance=0.1) -> int:
    """#returns the number of the gotten lines in lines"""

    line = np.zeros([2, 2], dtype=float)  # хранит 2 столбца с координатами х, у начальной и конечной точек отрезка
    pcross = np.array([0.0, 0.0])

    i = 1
    Nlines = 0
    A_prev, C_prev = 0.0, 0.0
    while i < Npnts:
        gap = tolerance
        i0 = i
        sums = np.zeros([5], dtype=float)
        while True:  # проход по всем точкам
            # Новый формат - столбцы х, у
            line[:, 0] = pnts[:2, i - 1]  # столбец координат начальной точки (х1 у1)
            line[:, 1] = pnts[:2, i]  # столбец координат конечной точки (х2 у2)
            A, C, q0, sums = line_approx_lsm(pnts, i - 1, i + 1, sums)
            byNpnts = 2

            while True:  # проход по найденному отрезку - поиск конца
                i += 1
                if i < Npnts and abs(A * pnts[0, i] - pnts[1, i] + C) / m.sqrt(
                        A ** 2 + 1) < gap:  # если есть следующая точка и tolerance не превышен
                    if not byNpnts % 2:
                        A = (pnts[1, i - byNpnts // 2] - pnts[1, i - byNpnts]) / (
                                    pnts[0, i - byNpnts // 2] - pnts[0, i - byNpnts])
                        C = pnts[1, i - byNpnts] - A * pnts[0, i - byNpnts]
                    byNpnts += 1
                else:
                    A, C, q0, sums = line_approx_lsm(pnts, i - byNpnts, i, sums)
                    while q0 > 0.0001:  # поиск оптимальной концевой точки, чтобы не забрать лишних следующих
                        # if mode == 'lsm':
                        #     A_opt, C_opt, q, sums = line_approx_lsm(pnts, i - byNpnts, i - 1, sums)
                        # elif mode == 'opt_lsm':
                        A_opt, C_opt, q, sums = line_approx_lsm(pnts, i - byNpnts, i, sums, True)

                        if q > q0:  # если увеличилось ср. отклонение, прерываем (дальше с последним оптимумом)
                            break
                        else:  # сохраняем текущий оптимум
                            i -= 1
                            byNpnts -= 1
                            A = A_opt
                            C = C_opt
                            q0 = q

                    # Работаем с полученными А и С
                    if Nlines > 0:  # если уже найден хотя бы один луч - ищем пересечение текущей прямой с ним

                        if A == A_prev:
                            pcross[0], pcross[1] = inf, inf  # прямые параллельны, в т.ч. и если обе вертикальные
                        else:
                            if isinf(A_prev):
                                pcross[0], pcross[1] = C_prev, A * C_prev + C  # вертикальная исходная
                            elif isinf(A):
                                pcross[0], pcross[1] = C, A_prev * C + C_prev  # вертикальная подставляемая
                            else:
                                pcross[0] = (C_prev - C) / (A - A_prev)
                                pcross[1] = A_prev * (C_prev - C) / (A - A_prev) + C_prev

                        if np.linalg.norm(pnts[:2, i - byNpnts] - pcross) > tolerance or \
                                m.isnan(pcross[0]) or m.isinf(pcross[0]):
                            if byNpnts <= 2:
                                pcross[0] = (pnts[0, i - 2] + A_prev * pnts[1, i - 2] - A_prev * C_prev) \
                                            / (A_prev ** 2 + 1)
                                pcross[1] = A_prev * pcross[0] + C_prev

                                line[0, 0] = pcross[0]
                                line[1, 0] = pcross[1]

                            else:
                                i = i0
                                gap *= 0.75
                                break
                        else:
                            line[0, 0] = pcross[0]
                            line[1, 0] = pcross[1]

                    else:  # если ещё не нашли линий - пересечение прямой с нормалью из первой точки датасета
                        A_prev, C_prev = calc_normal(pnts[:2, 0], A)
                        line[0, 0], line[1, 0] = calc_intersection(A_prev, C_prev, A, C)
                        # =========== ПОЛУЧАЕМ ЛУЧ, ИСХОДЯЩИЙ ИЗ ЭТОЙ ТОЧКИ (КОНЕЦ ПОКА НЕ ИЗВЕСТЕН) ==============

                    if i > Npnts - 1:  # если точка последняя - пересечение прямой с нормалью из этой точки
                        A_last, C_last = calc_normal(pnts[:2, Npnts - 1], A)
                        line[0, 1], line[1, 1] = calc_intersection(A, C, A_last, C_last)

                    break

            A_prev = A
            C_prev = C

            if i > i0:
                break
            else:
                continue

        lines[:, Nlines] = line[:, 0]
        Nlines += 1
        if i > Npnts - 1:
            lines[:, Nlines] = line[:, 1]

    return Nlines
