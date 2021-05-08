import numpy
from matplotlib import pyplot, cm

# температура плавдения
t0 = 1538

# начальная температура
t_initial = 1500

# теплопроводность
k1 = 0.5
k2 = 0.75
# теплоемкость
c1 = 2
c2 = 1.25
# плотность
rho1 = 1
rho2 = 2
# энтальпия
L = 1
# дельта
d = 0.15

c0 = L / (2 * d) + (c1 + c2) / 2

# источники
j_s = 1
j_n = 1
j_w = 1
j_e = 1


def heat_capacity(t):
    if t <= t0 - d:
        return c1
    elif t >= t0 + d:
        return c2
    elif t0 - d < t < t0:
        return (t0 * c1 - (t0 - d) * c0 - (c1 - c0) * t) / (t0 - (t0 - d))
    elif t0 + d > t > t0:
        return ((t0 + d) * c0 - t0 * c2 - (c0 - c2) * t) / (t0 + d - t0)
    elif t == t0:
        return c0


def heat_conductivity(t):
    if t <= t0 - d:
        return k1
    elif t0 - d < t < t0 + d:
        return ((t0 + d) * k1 - (t0 - d) * k2 - (k1 - k2) * t) / (2 * d)
    elif t >= t0 + d:
        return k2


def density(t):
    if t <= t0 - d:
        return rho1
    elif t0 - d < t < t0 + d:
        return ((t0 + d) * rho1 - (t0 - d) * rho2 - (rho1 - rho2) * t) / (2 * d)
    elif t >= t0 + d:
        return rho2


time_max = 10
dt = 1

nx = 20
ny = 20

lx = 1
ly = 1

dx = lx / nx
dy = ly / ny

u_current = numpy.zeros((ny, nx))
v_current = numpy.zeros((ny, nx))
t_current = numpy.zeros((ny, nx))

t_current[::] = t_initial

result = [t_current]


def F_w(time, i, j):
    if j == 0:
        return 0
    nb_temperature = (result[time - 1][i - 1][j] + result[time - 1][i][j]) / 2
    nb_u = (u_current[i - 1][j] + u_current[i][j]) / 2
    return density(nb_temperature) * nb_u * dy


def F_e(time, i, j):
    if j == ny - 1:
        return 0
    nb_temperature = (result[time - 1][i + 1][j] + result[time - 1][i][j]) / 2
    nb_u = (u_current[i + 1][j] + u_current[i][j]) / 2
    return density(nb_temperature) * nb_u * dy


def F_s(time, i, j):
    if i == 0:
        return 0
    nb_temperature = (result[time - 1][i][j - 1] + result[time - 1][i][j]) / 2
    nb_u = (u_current[i][j - 1] + u_current[i][j]) / 2
    return density(nb_temperature) * nb_u * dx


def F_n(time, i, j):
    if i == nx - 1:
        return 0
    nb_temperature = (result[time - 1][i][j + 1] + result[time - 1][i][j]) / 2
    nb_u = (u_current[i][j + 1] + u_current[i][j]) / 2
    return density(nb_temperature) * nb_u * dx


def D_w(time, i, j):
    nb_temperature = (result[time - 1][i - 1][j] + result[time - 1][i][j]) / 2
    return (heat_conductivity(nb_temperature) * dy) / dx


def D_e(time, i, j):
    nb_temperature = (result[time - 1][i + 1][j] + result[time - 1][i][j]) / 2
    return (heat_conductivity(nb_temperature) * dy) / dx


def D_s(time, i, j):
    nb_temperature = (result[time - 1][i][j - 1] + result[time - 1][i][j]) / 2
    return (heat_conductivity(nb_temperature) * dx) / dy


def D_n(time, i, j):
    nb_temperature = (result[time - 1][i][j + 1] + result[time - 1][i][j]) / 2
    return (heat_conductivity(nb_temperature) * dx) / dy


def a_e(time, i, j):
    if j == ny - 1:
        return 0
    temperature = result[time - 1][i][j]
    return D_e(time, i, j) + heat_capacity(temperature) * max(-F_e(time, i, j), 0)


def a_w(time, i, j):
    if j == 0:
        return 0
    temperature = result[time - 1][i][j]
    return D_w(time, i, j) + heat_capacity(temperature) * max(F_w(time, i, j), 0)


def a_n(time, i, j):
    if i == nx - 1:
        return 0
    temperature = result[time - 1][i][j]
    return D_n(time, i, j) + heat_capacity(temperature) * max(-F_n(time, i, j), 0)


def a_s(time, i, j):
    if i == 0:
        return 0
    temperature = result[time - 1][i][j]
    return D_s(time, i, j) + heat_capacity(temperature) * max(F_s(time, i, j), 0)


def a_p(time, i, j):
    temperature = result[time - 1][i][j]
    return (heat_capacity(temperature) * density(temperature) * dx * dy) / dt + a_e(time, i, j) + a_w(time, i, j) \
           + a_n(time, i, j) + a_s(time, i, j) + heat_capacity(temperature) * (F_e(time, i, j) - F_w(time, i, j)) \
           + heat_capacity(temperature) * (F_n(time, i, j) - F_s(time, i, j))





