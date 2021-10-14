# 非線形回帰モデル 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

n = 50

def true_func(x):
    z = 1 - 50 * x + 200 * x**2 - 300 * x**3 + 150 * x**4
    return z 

# 真の関数からデータ生成する処理
data = np.random.rand(n).astype(np.float32)
data = np.sort(data)
target = true_func(data)

# ノイズを加える処理
noise = 0.5 * np.random.randn(n) 
target = target  + noise

# ノイズ付きデータを描画する処理
plt.scatter(data, target)
plt.title('NonLinear Regression')

# モデルの作成と学習する処理
clf = LinearRegression()
data = data.reshape(-1,1) # n行1列に変形する
target = target.reshape(-1,1)
clf.fit(data, target)

# 学習させたモデルで予測する処理
p_lin = clf.predict(data)

plt.scatter(data, target, label='data')
plt.plot(data, p_lin, color='darkorange', marker='', linestyle='-', linewidth=1, markersize=6, label='linear regression')
plt.legend()
print(clf.score(data, target))

# モデルの作成と学習する処理
clf = KernelRidge(alpha=0.0002, kernel='rbf')
clf.fit(data, target)

# 学習させたモデルで予測する処理
p_kridge = clf.predict(data)

plt.scatter(data, target, color='blue', label='data')
plt.plot(data, p_kridge, color='orange', linestyle='-', linewidth=3, markersize=6, label='kernel ridge')
plt.legend()