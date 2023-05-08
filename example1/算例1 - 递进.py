import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import sciann as sn
from numpy import pi
from sciann.utils.math import diff, sign, exp, sin, sinh, cos, tanh


w = 0.1
i = 1
begin = time.time()
while w > 0.01:
    # 导入第一次权值
    x = sn.Variable('x')
    u = sn.Functional('u', x, 4 * [20], 'tanh')

    L1 = (-w * diff(u, x, order=2) + diff(u, x) - w * pi * pi * sin(pi * x) - pi * cos(pi * x))*0.1

    TOL = 0.001
    C1 = (1 - sign(x - TOL)) * u  # 左边界 x = 0
    C2 = (1 + sign(x - (1 - TOL))) * (u - 1)  # 右边界 x = 2

    m = sn.SciModel(x, [L1, C1, C2])
    x_data = np.linspace(0, 1, 500)
    m.load_weights('bodong-1D-100-2.hdf5')   # 加载模型
    weight_a = u.get_weights()     # 获取权值

    # 设置权值训练
    w = 0.7*w
    x = sn.Variable('x')
    u = sn.Functional('u', x, 4 * [20], 'tanh')
    u.set_weights(weight_a)

    L1 = (-w * diff(u, x, order=2) + diff(u, x) - w * pi * pi * sin(pi * x) - pi * cos(pi * x))*0.1

    TOL = 0.001
    C1 = (1 - sign(x - TOL)) * u  # 左边界 x = 0
    C2 = (1 + sign(x - (1 - TOL))) * (u - 1)  # 右边界 x = 2
    m = sn.SciModel(x, [L1, C1, C2])
    x_data = np.linspace(0, 1, 500)
    h = m.train(
        x_data,
        3 * ['zero'],
        learning_rate=0.005,
        # batch_size=100,
        epochs=5000,
        # adaptive_weights={'method': 'NTK', 'freq': 100},
        # shuffle=True,
        # stop_loss_value=1e-6,
        verbose=1)
    lossz = np.array(h.history['loss'])
    np.savez('{}'.format(i), lossz)
    plt.semilogy(h.history['loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    m.save_weights('bodong-1D-100-2.hdf5')
    i = i+1

# 获取权值
x = sn.Variable('x')
u = sn.Functional('u', x, 4 * [20], 'tanh')

L1 = (-w * diff(u, x, order=2) + diff(u, x) - w * pi * pi * sin(pi * x) - pi * cos(pi * x))*0.1

TOL = 0.001
C1 = (1 - sign(x - TOL)) * u  # 左边界 x = 0
C2 = (1 + sign(x - (1 - TOL))) * (u - 1)  # 右边界 x = 2

m = sn.SciModel(x, [L1, C1, C2])
x_data = np.linspace(0, 1, 500)
m.load_weights('bodong-1D-100-2.hdf5')
weight_b = u.get_weights()

# 设置权值训练
x = sn.Variable('x')
u = sn.Functional('u', x, 4 * [20], 'tanh')
u.set_weights(weight_b)
w = 0.01
L1 = (-w * diff(u, x, order=2) + diff(u, x) - w * pi * pi * sin(pi * x) - pi * cos(pi * x))*0.1

TOL = 0.001
C1 = (1 - sign(x - TOL)) * u  # 左边界 x = 0
C2 = (1 + sign(x - (1 - TOL))) * (u - 1)  # 右边界 x = 2

m = sn.SciModel(x, [L1, C1, C2])
x_data = np.linspace(0, 1, 500)

h = m.train(
    x_data,
    3 * ['zero'],
    learning_rate=0.005,
    # batch_size=100,
    epochs=15000,
    # adaptive_weights={'method': 'NTK', 'freq': 100},
    # shuffle=True,
    # stop_loss_value=1e-6,
    verbose=1)

x_test = np.linspace(0, 1, 200)
u_pred = u.eval(m, x_test)
# weight = m.get_weights()
time_cost = time.time() - begin
print('Training complete in {:.0f}m {:.0f}s'.format(time_cost // 60, time_cost % 60))
m.save_weights('w=0.001.hdf5')
lossz = np.array(h.history['loss'])
# lossL = np.array(history.history['mul_74_loss'])
# lossC1 = np.array(history.history['mul_75_loss'])
# lossC2 = np.array(history.history['mul_76_loss'])
# lossC3 = np.array(history.history['mul_77_loss'])
# lossC4 = np.array(history.history['mul_78_loss'])

np.savez('save1', lossz)
# data = np.load('save1.npz')
# print(data['arr_0'])
# print(data['arr_1'])
# print(data['arr_2'])
# print(data['arr_3'])
m.save_weights('suanli-1-dijin.hdf5')
plt.semilogy(h.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('loss.png')
plt.savefig('loss.eps')
plt.show()
# print(weight)
# print(u_pred)
# print(x_test)
fig = plt.figure(figsize=(6, 6))
plt.plot(x_test, u_pred)
plt.savefig('pinn图.png')
plt.show()
plt.show()

