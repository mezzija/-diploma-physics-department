import numpy
from matplotlib import pyplot

nx = 11
ny = 11
nt = 10
nit = 50

c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)

rho = 1
nu = .1
dt = .001
pr = 1
ra = 1
#  начальные условия
u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx))
b = numpy.zeros((ny, nx))
t = numpy.zeros((ny, nx))

# граничные условия
t[:, -1] = 1
t[:, 0] = 10


def up_t(prevT, prevU, prevV):
    tn = prevT.copy()
    un = prevU.copy()
    vn = prevV.copy()
    t[1:-1, 1:-1] = tn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (tn[1:-1, 1:-1] - tn[2:, 1:-1]) - \
                    vn[1:-1, 1:-1] * dt / dy * (tn[1:-1, 1:-1] - tn[1:-1, 2:]) + \
                    dt / dx ** 2 * (tn[2:, 1:-1] - 2 * tn[1:-1, 1:-1] + tn[0:-2, 1:-1]) + \
                    dt / dy ** 2 * (tn[1:-1, 2:] - 2 * tn[1:-1, 1:-1] + tn[1:-1, 0:-2])
    return t


def up_p(currentT, prevU, prevV, prevP):
    tn = currentT.copy()
    un = prevU.copy()
    vn = prevV.copy()
    pn = prevP.copy()

    p[1:-1, 1:-1] = ((pn[2:, 1:-1] + p[0:-2, 1:-1]) * dy ** 2 + (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dx ** 2) / (
                2 * (dx ** 2 + dy ** 2)) - \
                    rho * ((dx ** 2 * dy ** 2) / 2 * (dx ** 2 + dy ** 2)) * \
                    (1 / dt * ((un[2:, 1:-1] - un[0:-2, 1:-1]) / 2 * dx + (vn[1:-1, 2:] - vn[1:-1, 0:-2]) / 2 * dy) -
                     ((un[2:, 1:-1] - un[0:-2, 1:-1]) / 2 * dx) ** 2 - 2 * (un[1:-1, 2:] - un[1:-1, 0:-2]) / 2 * dy *
                     (vn[2:, 1:-1] - vn[0:-2, 1:-1]) / 2 * dx - ((vn[1:-1, 2:] - vn[1:-1, 0:-2]) / 2 * dy) ** 2) - \
                    pr * ra * ((dy ** 2 * dx ** 2) / 2 * (dy ** 2 + dx ** 2)) * (tn[1:-1, 2:] - tn[1:-1, 0:-2]) / 2 * dy
    return p


def up_u(currentT, currentP, prevU, prevV):
    tn = currentT.copy()
    pn = currentP.copy()
    un = prevU.copy()
    vn = prevV.copy()
    u[1:-1, 1:-1] = un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) - \
                    vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - \
                    dt / 2 * dx * (pn[2:, 1:-1] - p[0:-2, 1:-1]) + \
                    pr * nu * (dt / dx ** 2 * (un[2, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]) +
                               dt / dy ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])) + \
                    pr * ra * dt * tn[1:-1, 1 - 1]
    return u


def up_v(currentT, currentP, prevU, prevV):
    tn = currentT.copy()
    pn = currentP.copy()
    un = prevU.copy()
    vn = prevV.copy()
    v[1:-1, 1:-1] = vn[1:-1, 1:-1] - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - \
                    vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) + \
                    dt / 2 * dy * (pn[1:-1, 2:] - pn[1:-1, 0:-2]) + \
                    pr * nu * (dt / dx ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) +
                               dt / dy ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])) + \
                    pr * ra * dt * tn[1:-1, 1:-1]
    return v

resultT = numpy.zeros((ny, nx, nt))
resultT[:, :, 0] = up_t(t, u, v)

resultP = numpy.zeros((ny, nx, nt))
resultP[:, :, 0] = up_p(resultT[:, :, 0], u, v, p)

resultU = numpy.zeros((ny, nx, nt))
resultU[:, :, 0] = up_u(resultT[:, :, 0], resultP[:, :, 0], u, v)

resultV = numpy.zeros((ny, nx, nt))
resultV[:, :, 0] = up_v(resultT[:, :, 0], resultP[:, :, 0], u, v)


for t in range(1, nt):
    resultT[:, :, nt] = up_t(resultT[:, :, nt - 1], resultU[:, :, nt - 1], resultV[:, :, nt - 1])
    resultP[:, :, nt] = up_p(resultT[:, :, nt], resultU[:, :, nt - 1], resultV[:, :, nt - 1], resultP[:, :, nt - 1])
    resultU[:, :, nt] = up_u(resultT[:, :, nt], resultP[:, :, nt], resultV[:, :, nt - 1], resultP[:, :, nt - 1])
    resultV[:, :, nt] = up_v(resultT[:, :, nt], resultP[:, :, nt], resultV[:, :, nt - 1], resultP[:, :, nt - 1])
