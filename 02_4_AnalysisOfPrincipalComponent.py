#32次元のデータを2次元上に次元圧縮した際に、うまく判別できるかを確認。
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

cancer_df = pd.read_csv(r'C:\e_credential_report\breast_cancer\data.csv')

cancer_df['Unnamed: 32'].isnull().all()
cancer_df.drop('Unnamed: 32', axis=1, inplace=True)

def confusion_matrix_flatten(cm):
    tn, fp, fn, tp = cm.flatten()

    print('TP:', tp)
    print('FP:', fp)
    print('TN:', tn)
    print('FN:', fn)
    print()

    
# 目的変数の抽出処理
y = cancer_df.diagnosis.apply(lambda d: 1 if d == 'M' else 0)

# 説明変数の抽出処理
X = cancer_df.loc[:, 'radius_mean':]

# 学習用、検証用データに分割する処理
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 標準化処理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ロジスティック回帰で学習する処理
logistic = LogisticRegressionCV(cv=10, random_state=1, max_iter=10000)
logistic.fit(X_train_scaled, y_train)

# 検証結果
print('Train : {:.3f}'.format(logistic.score(X_train_scaled, y_train)))
print('Test : {:.3f}'.format(logistic.score(X_test_scaled, y_test)))
print()
cm = confusion_matrix(y_true=y_test, y_pred=logistic.predict(X_test_scaled))
confusion_matrix_flatten(cm)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# 寄与率
print('寄与率: {}'.format(pca.explained_variance_ratio_))
print('累積寄与率: {}'.format(pca.explained_variance_ratio_.sum()))

# 散布図にプロット
temp = pd.DataFrame(X_train_pca)
temp['Outcome'] = y_train.values
b = temp[temp['Outcome'] == 0]
m = temp[temp['Outcome'] == 1]
plt.scatter(x=b[0], y=b[1], marker='o') # 良性は○でマーク
plt.scatter(x=m[0], y=m[1], marker='^') # 悪性は△でマーク
plt.xlabel('PC1') # 第1主成分をx軸
plt.ylabel('PC2') # 第2主成分をy軸
