import numpy
from matplotlib import pyplot

time = 10

dt = 1

x_max = 10
y_max = 8

big_dx = 0.1
big_dy = 0.05

small_dx = big_dx / 2
small_dy = big_dy / 2

density = 7874
viscosity = 0.0000006

u = numpy.zeros((int(x_max / big_dx), int(y_max / big_dy)))
v = numpy.zeros((int(x_max / big_dx), int(y_max / big_dy)))

A_p = 1


def f_e(i, j, prev):
    return (density * prev[i + 1][j]) / big_dx


def f_w(i, j, prev):
    return (density * prev[i - 1][j]) / big_dx


def f_n(i, j, prev):
    return (density * prev[i][j + 1]) / big_dy


def f_s(i, j, prev):
    return (density * prev[i][j - 1]) / big_dy


def d_e():
    return viscosity / big_dx


def d_w():
    return viscosity / big_dx


def d_n():
    return viscosity / big_dy


def d_s():
    return viscosity / big_dy


def a_e(i, j, prev):
    return d_e() + (0 if f_e(i, j, prev) >= 0 else -f_e(i, j, prev))


def a_w(i, j, prev):
    return d_w() + (f_w(i, j, prev) if f_w(i, j, prev) >= 0 else 0)


def a_n(i, j, prev):
    return d_n() + (0 if f_n(i, j, prev) else -f_n(i, j, prev))


def a_s(i, j, prev):
    return d_s() + (f_s(i, j, prev) if f_s(i, j, prev) >= 0 else 0)


def a_p(i, j, prev):
    return density/dt + A_p + a_e(i, j, prev) + a_w(i, j, prev) + a_n(i, j, prev) + a_s(i, j, prev) + f_e(i, j, prev) \
           - f_w(i, j, prev) + f_n(i, j, prev) - f_s(i, j, prev)



