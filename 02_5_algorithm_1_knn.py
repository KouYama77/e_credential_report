#人口データと分類結果をプロットしてください。
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

from scipy import stats

# 訓練データ作成処理
x0 = np.random.normal(size=50).reshape(-1, 2) - 1
x1 = np.random.normal(size=50).reshape(-1, 2) + 1
X_train = np.concatenate([x0, x1])
y_train = np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int64)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)

# 距離の合計値を取得する処理
def distance(x1, x2):
    return np.sum((x1 - x2)**2, axis=1)

# 予測処理
def knc_predict(n_neighbors, X_train, y_train, X_test):
    y_pred = np.empty(len(X_test), dtype=y_train.dtype)
    for i, x in enumerate(X_test):
        distances = distance(x, X_train)
        nearest_index = distances.argsort()[:n_neighbors]
        mode, _ = stats.mode(y_train[nearest_index])
        y_pred[i] = mode
    
    return y_pred

# プロット処理
def plt_result(X_train, y_train, y_pred):
    x0, x1 = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    x = np.array([x0, x1]).reshape(2, -1).T
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.contourf(x0, x1, y_pred.reshape(100, 100).astype(dtype=np.float64), alpha=0.2, levels=np.linspace(0, 1, 3))

n_neighbors = 3

x0, x1 = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
X_test = np.array([x0, x1]).reshape(2, -1).T

y_pred = knc_predict(n_neighbors, X_train, y_train, X_test)
plt_result(X_train, y_train, y_pred)
