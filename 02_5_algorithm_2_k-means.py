# k-平均法(k-means)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs as mb

def distance(x1, x2):
    return np.sum((x1 - x2)**2, axis=1)

def plt_result(X_train, centers, xs):
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_pred, cmap='spring')
    plt.scatter(centers[:, 0], centers[:, 1], s=200, marker='X', lw=2, c='black', edgecolor='white')
    pred = np.empty(len(xs), dtype=int)
    for i, x in enumerate(xs):
        d = distance(x, centers)
        pred[i] = np.argmin(d)
    plt.contourf(xs0, xs1, pred.reshape(100, 100), alpha=0.2, cmap='spring')

clf = KMeans(n_clusters = 3)

dataset = mb(centers = 3)
features = np.array(dataset[0])
pred = clf.fit_predict(features)

# 訓練データ作成処理
x0 = np.random.normal(size=50).reshape(-1, 2) - 1
x1 = np.random.normal(size=50).reshape(-1, 2) + 1
X_train = np.concatenate([x0, x1])
y_train = np.concatenate([np.zeros(25), np.ones(25)]).astype(np.int64)

plt.scatter(X_train[:, 0], X_train[:, 1])

n_clusters = 3
iter_max = 100000

centers = X_train[np.random.choice(len(X_train), n_clusters, replace=False)]

for _ in range(iter_max):
    prev_centers = np.copy(centers)
    D = np.zeros((len(X_train), n_clusters))
    for i, x in enumerate(X_train):
        D[i] = distance(x, centers)
    cluster_index = np.argmin(D, axis=1)
    for k in range(n_clusters):
        index_k = cluster_index == k
        centers[k] = np.mean(X_train[index_k], axis=0)
    if np.allclose(prev_centers, centers):
        break
        
y_pred = np.empty(len(X_train), dtype=int)

for i, x in enumerate(X_train):
    d = distance(x, centers)
    y_pred[i] = np.argmin(d)

xs0, xs1 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
xs = np.array([xs0, xs1]).reshape(2, -1).T

plt_result(X_train, centers, xs)