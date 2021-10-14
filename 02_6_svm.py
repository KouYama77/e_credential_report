# SVM(�T�|�[�g�x�N�^�[�}�V��)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = load_iris()

dataset = pd.DataFrame(data = iris['data'], columns = iris['feature_names'])
dataset['species'] = iris['target']

# �ړI�ϐ�(Y)�A�����ϐ�(X)
y = np.array(dataset['species'])
X = np.array(dataset[iris['feature_names']])

# �f�[�^�̕���
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = SVC(gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('�P���f�[�^�F')
print(Y_test[:10])
print('���؃f�[�^�F')
print(y_pred[:10])
