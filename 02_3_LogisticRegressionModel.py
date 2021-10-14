#年齢が30歳で男の乗客は生き残れるか？
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# データ前処理（欠損値・外れ値・One-Hot-Encoding・標準化）
def preprocess(df):
    
    # 欠損値の処理
    df['Age'] = df['Age'].fillna(df['Age'].median())              # 中央値
    df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode())          # 最頻値
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()) # 最頻値
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())           # 中央値
    
    # 外れ値の処理
    df['Parch'] = np.where(df['Parch']==9, 6, df['Parch'])
    
    # ダミー変数
    df_dummies = pd.get_dummies(df, columns=['Sex', 'Pclass', 'SibSp', 
                                             'Parch', 'Embarked'])
    # データを標準化（平均 0, 分散 1）
    scaler = StandardScaler()
    
    df_dummies['Age_scale'] = scaler.fit_transform(df_dummies.loc[:, ['Age']])
    df_dummies['Fare_scale'] = scaler.fit_transform(df_dummies.loc[:, ['Fare']])
    
    return df_dummies
        
# 目的変数・説明変数を作成
def split_data(flag):
    
    # 学習用データの場合
    if flag == 'train':
        # 説明変数
        X = df_dummies.drop(drop_cols[flag], axis=1)
        # 目的変数
        y = df_dummies['Survived'].values
        
        return X, y
    else:
        # 説明変数
        X = df_test_dummies.drop(drop_cols[flag], axis=1)
        
        return X
    
# 予測モデルを作成
def model(X, y):
    
    clf.fit(X, y)
    coef = clf.coef_
    
# 予測
def predict(X):
    
    # 予測値を計算
    y = clf.predict(X)
    return y

# 提出用データをエクスポート（id・予測値）
def export_data(predict, submit_file):
    
    # 予測結果とスコアをテストデータに追加
    df_test['Survived'] = predict
    
    # 提出用データをエクスポート（id・予測値）
    # CSV ファイルを出力
    df_test.loc[:, ['PassengerId', 'Survived']].to_csv(
            submit_file, index=False, header=True)
    
    print(df_test.loc[:, ['PassengerId', 'Survived']])


train_file = r'C:\e_credential_report\titanic\train.csv'   # 学習用データ
test_file = r'C:\e_credential_report\titanic\test.csv'     # テスト用データ
result_file = r'C:\e_credential_report\titanic\result.csv' # 提出用データ

df = pd.read_csv(train_file, index_col=0, encoding='UTF-8')
df_test = pd.read_csv(test_file, index_col=None, encoding='UTF-8')

test_size = 0.2 # テスト用データの割合
c = 1.0         # 正則化のパラメータ（デフォルト 1.0）

# 分類レポートのラベル
targets = ['死亡', '生存']  ## 0 : 死亡, 1 : 生存

drop_cols = {'train' : ['Survived', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin'], 
             'test' : ['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin']}

# データを読み込む
df = pd.read_csv(train_file, index_col=0, encoding='UTF-8')
df_test = pd.read_csv(test_file, index_col=None, encoding='UTF-8')

# 学習用データ・テスト用データ
df_dummies = preprocess(df)
df_test_dummies = preprocess(df_test)

# ロジスティック回帰のモデル
clf = linear_model.LogisticRegression(C=c, solver='liblinear', 
                                           random_state=0)
# 回帰係数
coef = []

# 学習用データより目的変数・説明変数を作成
(X, y) = split_data('train')

# 学習用・検証用データに分割
(X_train, X_valid, y_train, y_valid) = train_test_split(
        X, y, test_size=test_size, random_state=0)

# 予測モデルを作成
model(X_train, y_train)

# 予測
y_valid_pred = predict(X_valid)

# テスト用データを予測
predict_result = predict(split_data('test'))

# 予測結果をエクスポート
export_data(predict_result, result_file)

