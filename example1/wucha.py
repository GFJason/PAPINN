import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, exp, tanh, sin, cos

x = sn.Variable('x')
u = sn.Functional('u', x, 4*[20], 'tanh')

w = 0.01
L1 = (-w*diff(u, x, order=2) + diff(u, x) - w*pi*pi*sin(pi*x) - pi*cos(pi*x))

TOL = 0
C1 = (1-sign(x - TOL)) * u  # 左边界 x = 0
C2 = (1+sign(x - (1-TOL))) * (u - 1)   # 右边界 x = 2

m = sn.SciModel(x, [L1, C1, C2])
x_data = np.linspace(0, 1, 500)

m.load_weights('bodong-1D-100-2.hdf5')
Nx = 200
x_test = np.linspace(0, 1, Nx)
u_pred = u.eval(m, x_test)
z = np.sin(pi*x_test)+((np.exp(x_test/w)-1)/(np.exp(1/w)-1))

a = abs(z-u_pred)
print(a)
Errorav = sum(a) / len(a)
print(Errorav)
# 最大误差
Errormax = max(a)
print(Errormax)
# print('Error=', sum(err)/Nx)
# v
fig = plt.figure(figsize=(6, 6))
plt.plot(x_test, u_pred)
plt.ylim(0, 1.6)
plt.plot(x_test, z, color='red', linestyle=':')
plt.xlabel('x')
plt.ylabel('U')
plt.savefig('pinn图.png')
plt.savefig('pinn图.eps')
plt.show()


# Error V
fig,ax = plt.subplots(figsize=(6, 6))
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0,0))
ax.yaxis.set_major_formatter(formatter)
# fig = plt.figure(figsize=(6, 6))
plt.plot(x_test, abs(z-u_pred))
plt.xlabel('x')
plt.ylabel('Error U')
plt.savefig('误差图.jpg')
plt.savefig('误差图.eps')
plt.show()
