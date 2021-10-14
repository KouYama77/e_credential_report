# 課題　部屋数が4で犯罪率が0.3の物件はいくらになるか？
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ボストンデータをインポートする処理
boston = load_boston()

#　説明変数を変換する処理
df = pd.DataFrame(data=boston.data, columns = boston.feature_names)

# 目的変数を追加する処理
df['PRICE'] = np.array(boston.target)

# 特徴量と目的変数を分ける処理
df_X = df.drop(['PRICE'], axis=1)
df_y = df['PRICE']

# 特徴量を標準化する処理
scaler = StandardScaler()
df_Xs = scaler.fit_transform(df_X)
df_Xs = pd.DataFrame(df_Xs, columns=boston.feature_names)

# 学習用、検証用データに分割する処理
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size = 0.3, random_state = 2021)

# オブジェクト生成処理
model = LinearRegression()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# 学習用、検証用データで平均二乗誤差を出力する処理
print('Mean Squared Error Train : %.3f, Test : %.3f' % (mean_squared_error(y_train, pred_train), mean_squared_error(y_test, pred_test)))
