# �ۑ�@��������4�Ŕƍߗ���0.3�̕����͂�����ɂȂ邩�H
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# �{�X�g���f�[�^���C���|�[�g���鏈��
boston = load_boston()

#�@�����ϐ���ϊ����鏈��
df = pd.DataFrame(data=boston.data, columns = boston.feature_names)

# �ړI�ϐ���ǉ����鏈��
df['PRICE'] = np.array(boston.target)

# �����ʂƖړI�ϐ��𕪂��鏈��
df_X = df.drop(['PRICE'], axis=1)
df_y = df['PRICE']

# �����ʂ�W�������鏈��
scaler = StandardScaler()
df_Xs = scaler.fit_transform(df_X)
df_Xs = pd.DataFrame(df_Xs, columns=boston.feature_names)

# �w�K�p�A���ؗp�f�[�^�ɕ������鏈��
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size = 0.3, random_state = 2021)

# �I�u�W�F�N�g��������
model = LinearRegression()
model.fit(X_train, y_train)
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# �w�K�p�A���ؗp�f�[�^�ŕ��ϓ��덷���o�͂��鏈��
print('Mean Squared Error Train : %.3f, Test : %.3f' % (mean_squared_error(y_train, pred_train), mean_squared_error(y_test, pred_test)))
