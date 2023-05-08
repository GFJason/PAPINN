# 导入必须的包
import matplotlib.pyplot as plt
import numpy as np
# -----------  打开txt文件   ----------
s1= np.load('1.npz')
s2 = np.load('2.npz')
s3 = np.load('3.npz')
s4 = np.load('4.npz')
s5 = np.load('5.npz')
s6 = np.load('6.npz')
s7 = np.load('7.npz')
s8 = np.load('8.npz')
s9 = np.load('9.npz')
s10 = np.load('10.npz')
s11 = np.load('11.npz')
L4 = np.load('save1.npz')
a1 = s1['arr_0']
a2 = s2['arr_0']
a3 = s3['arr_0']
a4 = s4['arr_0']
a5 = s5['arr_0']
a6 = s6['arr_0']
a7 = s7['arr_0']
a8 = s8['arr_0']
a9 = s9['arr_0']
a10 = s10['arr_0']
a11 = s11['arr_0']
L = L4['arr_0']
# print(a.shape)
d = np.concatenate((a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, L), axis=0)
print(d)
# -----------  逐行读取文件内的数据  ------------
'''
txt文件的数值为y轴的数据
所以x要根据y的个数有序生成
'''
# ------ x轴数据有序生成150个（根据自己的横坐标范围自己修改范围）  ----
x = np.linspace(0, 115, 115)

# ----------  新建一个空的列表，用于存储上一步逐行读取的data  ------------
# ---------------    输出图    ----------------------
# ---------   可以理解为在图上加载x和y的数据   label为关于x和y曲线的标签------------
plt.semilogy(x, d)
# ---------   x轴的小标题   -------------
plt.xlabel('Epoch')
# ---------   y轴的小标题   -------------
plt.ylabel('Loss')
# ---------   整个图的标题  ----------
plt.title('yolov4-loss')
# plt.legend()
plt.show()
