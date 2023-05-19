import pandas as pd
import math
import numpy as np
from math import sqrt
from time import time
from scipy.stats import pearsonr, spearmanr
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from scipy.optimize import curve_fit, leastsq
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# 数据导入与预处理
y = pd.read_excel('./SIQAD/DMOS.xlsx')['score'].values
data = np.load('*.npy')
# 将数据转化为0,1正态分布
scale = StandardScaler()
data = scale.fit_transform(data)
# 去除nan值
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
data = imr.fit_transform(data)

y_ = np.empty(shape=[0, math.ceil(y.shape[0]*0.2)])
for j in range(5):
    X = np.empty(shape=[0, 517])
    for i in range(y.shape[0]):
        X = np.row_stack((X, data[5 * i + j]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练SVR模型
    grid_search = SVR(kernel='rbf', C=1000, epsilon=3, gamma=0.001)
    # grid_search = GridSearchCV(SVR(kernel='rbf', epsilon=2, C=100), cv=5,
                               # param_grid={"gamma": np.logspace(-4,-2,10)})
    grid_result = grid_search.fit(X_train, y_train)
    # print(grid_result.best_params_)
    y_svr = grid_result.predict(X_test)
    y_ = np.row_stack((y_, y_svr))
y_weighted = pow(y_[0], 0.15) * pow(y_[1], 0.25) * pow(y_[2], 0.35) * pow(y_[3], 0.20) * pow(y_[4], 0.05)


# 评价指标
plcc = pearsonr(y_test, y_weighted)[0]
srcc = spearmanr(y_test, y_weighted)[0]
rmse = sqrt(mean_squared_error(y_test, y_weighted))
print('plcc = %.4f' % plcc)
print('srcc = %.4f' % srcc)
print('rmse = %.4f' % rmse)


# 曲线映射
def func(x, β1, β2, β3, β4, β5):
    q = β1 * (1/2 - 1/(1 + np.exp(β2*(x-β3)))) + β4*x + β5
    return q

# 非线性最小二乘法拟合
popt, pcov = curve_fit(func, y_weighted, y_test, maxfev=500000)
# 获取popt里面的拟合系数
β1 = popt[0]
β2 = popt[1]
β3 = popt[2]
β4 = popt[3]
β5 = popt[4]
# 拟合y值
yvals = func(y_weighted,β1,β2,β3,β4,β5)

# # 绘图
# plt.figure(figsize=(14,8))
# plot1 = plt.plot(y_weighted, y_test, '*', label = 'original values')
# plot2 = plt.plot(y_weighted, yvals, 'r', label = 'curvefit values')
# plt.xlabel('Objective Scores')
# plt.ylabel('MOS')
# plt.legend(loc=4)   # 制定legend位置在右下角
# plt.title('curve_fit')
# # plt.savefig('E:/hyx/projects/SCIQA/output/_'+str(k+1)+'.jpg')
# plt.show()

plcc = pearsonr(y_test, yvals)[0]
srcc = spearmanr(y_test, yvals)[0]
rmse = sqrt(mean_squared_error(y_test, yvals))
print('%.4f' % plcc)
print('%.4f' % srcc)
print('%.4f' % rmse)


# # 寻找最优kernal
# kernels = ['linear','poly','rbf','sigmoid']
# eps = 5
# for kernel in kernels:
#     svr = SVR(kernel=kernel,epsilon=eps).fit(train_input, train_data)
#     y_svr = svr.predict(test_input)
#     print("kernel: {}".format(kernel))
#     perc_within_eps = 100 * np.sum(test_data - y_svr < eps) / len(test_data)
#     print("Percentage within Epsilon = {:,.2f}%".format(perc_within_eps))

# # 寻找最优C
# import math
# # C_range = np.logspace(-1,0,100)
# C_range = np.linspace(0.001,1,100)
# eps = 10
# max_perc = 0.5
# predicts = []
# for c in C_range:
#     svr = SVR(kernel='sigmoid',epsilon=eps,C=c).fit(train_input, train_data)#gramma无影响
#     y_svr = svr.predict(test_input)
#     perc_within_eps = 100 * np.sum(test_data - y_svr < eps) / len(test_data)
#     max_perc = max(max_perc, perc_within_eps)
#     predicts.append(perc_within_eps)
# print("Max percentage within Epsilon = {:,.2f}%".format(max_perc))
# plt.figure()
# plt.plot(C_range,predicts,c="blue")
# plt.xlabel('c')
# plt.ylabel('predict')
# plt.show()

# # 寻找最优gamma
# # gamma_range = np.logspace(-2,0,1000)
# gamma_range = np.linspace(1,1000,1000)
# eps = 10
# max_perc = 0.5
# predicts = []
# for gamma in gamma_range:
#     svr = SVR(kernel='sigmoid', epsilon=eps, gamma=gamma, C=0.001).fit(train_input, train_data)
#     y_svr = svr.predict(test_input)
#     perc_within_eps = 100 * np.sum(test_data - y_svr < eps) / len(test_data)
#     max_perc = max(max_perc,perc_within_eps)
#     predicts.append(perc_within_eps)
# print("Max percentage within Epsilon = {:,.2f}%".format(max_perc))
# plt.figure()
# plt.plot(gamma_range,predicts,c="red")
# plt.xlabel('gamma')
# plt.ylabel('predict')
# plt.show()



