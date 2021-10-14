#�N�30�΂Œj�̏�q�͐����c��邩�H
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# �f�[�^�O�����i�����l�E�O��l�EOne-Hot-Encoding�E�W�����j
def preprocess(df):
    
    # �����l�̏���
    df['Age'] = df['Age'].fillna(df['Age'].median())              # �����l
    df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode())          # �ŕp�l
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()) # �ŕp�l
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())           # �����l
    
    # �O��l�̏���
    df['Parch'] = np.where(df['Parch']==9, 6, df['Parch'])
    
    # �_�~�[�ϐ�
    df_dummies = pd.get_dummies(df, columns=['Sex', 'Pclass', 'SibSp', 
                                             'Parch', 'Embarked'])
    # �f�[�^��W�����i���� 0, ���U 1�j
    scaler = StandardScaler()
    
    df_dummies['Age_scale'] = scaler.fit_transform(df_dummies.loc[:, ['Age']])
    df_dummies['Fare_scale'] = scaler.fit_transform(df_dummies.loc[:, ['Fare']])
    
    return df_dummies
        
# �ړI�ϐ��E�����ϐ����쐬
def split_data(flag):
    
    # �w�K�p�f�[�^�̏ꍇ
    if flag == 'train':
        # �����ϐ�
        X = df_dummies.drop(drop_cols[flag], axis=1)
        # �ړI�ϐ�
        y = df_dummies['Survived'].values
        
        return X, y
    else:
        # �����ϐ�
        X = df_test_dummies.drop(drop_cols[flag], axis=1)
        
        return X
    
# �\�����f�����쐬
def model(X, y):
    
    clf.fit(X, y)
    coef = clf.coef_
    
# �\��
def predict(X):
    
    # �\���l���v�Z
    y = clf.predict(X)
    return y

# ��o�p�f�[�^���G�N�X�|�[�g�iid�E�\���l�j
def export_data(predict, submit_file):
    
    # �\�����ʂƃX�R�A���e�X�g�f�[�^�ɒǉ�
    df_test['Survived'] = predict
    
    # ��o�p�f�[�^���G�N�X�|�[�g�iid�E�\���l�j
    # CSV �t�@�C�����o��
    df_test.loc[:, ['PassengerId', 'Survived']].to_csv(
            submit_file, index=False, header=True)
    
    print(df_test.loc[:, ['PassengerId', 'Survived']])


train_file = r'C:\e_credential_report\titanic\train.csv'   # �w�K�p�f�[�^
test_file = r'C:\e_credential_report\titanic\test.csv'     # �e�X�g�p�f�[�^
result_file = r'C:\e_credential_report\titanic\result.csv' # ��o�p�f�[�^

df = pd.read_csv(train_file, index_col=0, encoding='UTF-8')
df_test = pd.read_csv(test_file, index_col=None, encoding='UTF-8')

test_size = 0.2 # �e�X�g�p�f�[�^�̊���
c = 1.0         # �������̃p�����[�^�i�f�t�H���g 1.0�j

# ���ރ��|�[�g�̃��x��
targets = ['���S', '����']  ## 0 : ���S, 1 : ����

drop_cols = {'train' : ['Survived', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin'], 
             'test' : ['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin']}

# �f�[�^��ǂݍ���
df = pd.read_csv(train_file, index_col=0, encoding='UTF-8')
df_test = pd.read_csv(test_file, index_col=None, encoding='UTF-8')

# �w�K�p�f�[�^�E�e�X�g�p�f�[�^
df_dummies = preprocess(df)
df_test_dummies = preprocess(df_test)

# ���W�X�e�B�b�N��A�̃��f��
clf = linear_model.LogisticRegression(C=c, solver='liblinear', 
                                           random_state=0)
# ��A�W��
coef = []

# �w�K�p�f�[�^���ړI�ϐ��E�����ϐ����쐬
(X, y) = split_data('train')

# �w�K�p�E���ؗp�f�[�^�ɕ���
(X_train, X_valid, y_train, y_valid) = train_test_split(
        X, y, test_size=test_size, random_state=0)

# �\�����f�����쐬
model(X_train, y_train)

# �\��
y_valid_pred = predict(X_valid)

# �e�X�g�p�f�[�^��\��
predict_result = predict(split_data('test'))

# �\�����ʂ��G�N�X�|�[�g
export_data(predict_result, result_file)

