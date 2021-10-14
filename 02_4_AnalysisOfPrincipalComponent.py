#32�����̃f�[�^��2������Ɏ������k�����ۂɁA���܂����ʂł��邩���m�F�B
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

    
# �ړI�ϐ��̒��o����
y = cancer_df.diagnosis.apply(lambda d: 1 if d == 'M' else 0)

# �����ϐ��̒��o����
X = cancer_df.loc[:, 'radius_mean':]

# �w�K�p�A���ؗp�f�[�^�ɕ������鏈��
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# �W��������
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ���W�X�e�B�b�N��A�Ŋw�K���鏈��
logistic = LogisticRegressionCV(cv=10, random_state=1, max_iter=10000)
logistic.fit(X_train_scaled, y_train)

# ���،���
print('Train : {:.3f}'.format(logistic.score(X_train_scaled, y_train)))
print('Test : {:.3f}'.format(logistic.score(X_test_scaled, y_test)))
print()
cm = confusion_matrix(y_true=y_test, y_pred=logistic.predict(X_test_scaled))
confusion_matrix_flatten(cm)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# ��^��
print('��^��: {}'.format(pca.explained_variance_ratio_))
print('�ݐϊ�^��: {}'.format(pca.explained_variance_ratio_.sum()))

# �U�z�}�Ƀv���b�g
temp = pd.DataFrame(X_train_pca)
temp['Outcome'] = y_train.values
b = temp[temp['Outcome'] == 0]
m = temp[temp['Outcome'] == 1]
plt.scatter(x=b[0], y=b[1], marker='o') # �ǐ��́��Ń}�[�N
plt.scatter(x=m[0], y=m[1], marker='^') # �����́��Ń}�[�N
plt.xlabel('PC1') # ��1�听����x��
plt.ylabel('PC2') # ��2�听����y��
