import torch
import torch.nn as nn
import tushare as ts
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 解决中文显示乱码问题
plt.rcParams['axes.unicode_minus'] = False
from sklearn.datasets import make_classification, make_circles, make_regression
from sklearn.model_selection import KFold
import sklearn.neural_network as net
import sklearn.linear_model as LM
from scipy.stats import multivariate_normal
from sklearn import svm
pro = ts.pro_api('818670fa68bc204c217143cdb75efeae1986031841ff8ca2c6a855bd')
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib as mpl
import csv
from matplotlib import cm
from scipy.stats import bartlett
import numpy.linalg as nlg
import warnings
warnings.filterwarnings("ignore")
label_encoder = LabelEncoder()
pd.set_option('mode.chained_assignment', None)

getdata = pd.read_excel('Data/Q1Data.xlsx')
getdata = getdata.set_index('小类')
print(getdata.head())

scaler =MinMaxScaler()
columns = getdata.columns[:]
scaler.fit(getdata[columns].values)
getdata[columns]= scaler.transform(getdata[columns].values)


factors=getdata.values
fa = FactorAnalyzer(n_factors=5,rotation=None)
fa.fit(factors)
ev, v = fa.get_eigenvalues()
plt.figure(figsize=(30, 8))
plt.scatter(range(1, factors.shape[1] + 1), ev,c='r',linewidth=10)
plt.plot(range(1, factors.shape[1] + 1), ev,c='b',linewidth=4)
plt.xlabel('factor')
plt.ylabel('Value')
plt.yticks([0.7,1.7,2.7,3.7,4.7,5.7,6.7],
           [r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$',r'$7$'])


plt.axhline(y=0.7 ,color='r', linestyle='--')
# 对纵坐标

plt.grid()
plt.show()


df2_corr = getdata.corr()

fig = plt.figure()
plt.figure(figsize=(10, 8))

sns.heatmap(df2_corr, cmap='coolwarm', fmt=".2f")
plt.xticks(rotation=45,fontsize=7)
plt.yticks(rotation=45,fontsize=7)
plt.show()

factors=getdata.drop(columns=['SOSA'])
y=getdata['SOSA']

# 随机森林
X_train, X_test, y_train, y_test = train_test_split(factors, y, test_size=0.1, shuffle=False)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train.values.ravel())
y_rf_pred = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_rf_pred))
win_ratio_rf = np.mean(np.sign(y_rf_pred) == np.sign(y_test.values))
print('Random Forest RMSE: ', rmse_rf)
print('Random Forest Win Ratio: ', win_ratio_rf)


feature_importances = rf_model.feature_importances_
plt.figure(figsize=(20, 6))
plt.barh(factors.columns, feature_importances, color=cm.rainbow(np.linspace(0, 0.5, len(factors.columns))),alpha=0.8)
plt.xlabel('因子重要性')
plt.ylabel('因子')
plt.title('随机森林模型因子重要性')
plt.show()

for factors, importance in zip(factors.columns, feature_importances):
    print(factors, importance)

AttentionQuery=pd.read_excel('AttentionQuery.xlsx')
AttentionQuery=AttentionQuery.iloc[:,0:7]
AttentionQuery.columns=['Browser', 'Popularity and Accessibility', 'Gender Equity','Sustainability', 'Inclusivity', 'Relevance and Innovation','Safety and Fair Play']

SumTrends = AttentionQuery.iloc[55:61, 0:7]
SumTrendsList = SumTrends.iloc[:, 1:7]
SumTrends = pd.DataFrame(SumTrendsList,
                         columns=['Popularity and Accessibility', 'Gender Equity', 'Sustainability', 'Inclusivity',
                                  'Relevance and Innovation', 'Safety and Fair Play'])

dataLenth = 6
angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)

# 创建2x3的子图布局，指定极坐标投影
fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': 'polar'})

# 将DataFrame的值转换为列表
SumTrendsList = SumTrends.iloc[:, :].values.tolist()

nameList = ['Baseball', 'Softball', 'Karate', 'Swimming', 'Basketball', 'Jumping']
colorList = ['r', 'g', 'b', 'y', 'c', 'm']
# 遍历每个项目和对应的子图
for i, (ax, project) in enumerate(zip(axs.ravel(), SumTrendsList)):
    # 确保data和angles长度一致
    data = np.concatenate((project, [project[0]]))
    angles = np.concatenate((angles[:6], [angles[0]]))

    # 在极坐标中绘制数据
    ax.plot(angles, data, color=colorList[i], marker='o')

    ax.set_title(nameList[i])  # 设置子图标题

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()
SumTrends = AttentionQuery.iloc[55:61, 0:7]


# 定义评估标准
criteria = ["Popularity and Accessibility", "Gender Equity", "Sustainability","Inclusivity", "Relevance and Innovation", "Safety and Fair Play"]

# 假设的权重
weights = np.array([0.25, 0.2, 0.15, 0.15, 0.2, 0.05])

# 构建判断矩阵（这里需要专家打分或者数据来填充）
judgment_matrix = np.array([
    [1,     1/0.8,  1/0.5,  1/0.6,  1/0.7,  1/0.9],
    [0.8,       1,  1/0.6,  1/0.7,  1/0.8,      1],
    [0.5,     0.6,      1,  1/1.2,  1/1.5,    1/2],
    [0.6,     0.7,    1.2,      1,  1/1.2,  1/1.5],
    [0.7,     0.8,    1.5,    1.2,      1,  1/1.3],
    [0.9,       1,      2,    1.5,    1.3,      1]
])
# 可视化
import seaborn as sns
# 更改大小
plt.figure(figsize=(10, 8))

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(judgment_matrix, annot=True, xticklabels=criteria, yticklabels=criteria, cmap=cmap)
# 不显示横坐标
plt.xticks([])
plt.show()


import numpy as np
from scipy.linalg import eig


# 计算权重
eigenvalues, eigenvectors = eig(judgment_matrix)
max_index = np.argmax(eigenvalues)
max_eigenvalue = eigenvalues[max_index].real
weights_normalized = np.abs(eigenvectors[:, max_index].real)
final_weights = weights_normalized / weights_normalized.sum()

# 一致性检验
CI = (max_eigenvalue - len(criteria)) / (len(criteria) - 1)
RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32]  # 随机一致性指数，取决于矩阵的阶数
CR = CI / RI[len(criteria) - 1]

print("Final Weights:", final_weights)
print("Consistency Ratio:", CR)

# 将权重用饼状图表示
plt.figure(figsize=(10, 10))
colors=['yellowgreen','gold','lightskyblue','lightcoral','pink','silver']
plt.pie(final_weights, labels=criteria, autopct='%1.1f%%', colors=colors)
plt.show()

SumTrends
SumTrendsList=SumTrends.iloc[:,1:7].values.tolist()

# 假设这是一些SDE的数据
sde_data = {
    "Baseball":SumTrendsList[0],
    "Softball":SumTrendsList[1],
    "Karate":SumTrendsList[2],
    "Swimming":SumTrendsList[3],
    "Basketball": SumTrendsList[4],
    "Jumping": SumTrendsList[5]
}

# 评估每个SDE
for sde, scores in sde_data.items():
    sde_score = np.dot(final_weights, scores)
    print(f"{sde}: Score = {sde_score}")

# 将Score排序并可视化
sorted_sde_data = {k: np.dot(final_weights, v) for k, v in sde_data.items()}
sorted_sde_data = dict(sorted(sorted_sde_data.items(), key=lambda x: x[1], reverse=True)
)
plt.figure(figsize=(30, 10))
# 用不同的颜色
plt.bar(sorted_sde_data.keys(), sorted_sde_data.values(), color=['red', 'blue', 'green', 'purple', 'orange', 'yellow'], alpha=0.5)
plt.xticks(rotation=45)
plt.xlabel("SDE")
plt.ylabel("Score")
plt.title("SDE Scores")
plt.show()

NewlyAddedAttention = pd.read_excel('Q4Data.xlsx')
NewlyAddedAttention.columns = ['Project', 'Popularity and Accessibility', 'Gender Equity', 'Sustainability',
                               'Inclusivity', 'Relevance and Innovation', 'Safety and Fair Play']
NewlyAddedAttention = NewlyAddedAttention.iloc[:, 1:7]

SumTrends = pd.DataFrame(NewlyAddedAttention,
                         columns=['Popularity and Accessibility', 'Gender Equity', 'Sustainability', 'Inclusivity',
                                  'Relevance and Innovation', 'Safety and Fair Play'])

dataLenth = 6
angles = np.linspace(0, 2 * np.pi, dataLenth, endpoint=False)

# 创建2x3的子图布局，指定极坐标投影
fig, axs = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': 'polar'})

# 将DataFrame的值转换为列表
SumTrendsList = SumTrends.iloc[:, :].values.tolist()

nameList = ["e-sports", "cricket", "steeplechase", "surfing", "Australianfootball", "Dragonboatrace"]
colorList = ['r', 'g', 'b', 'y', 'c', 'm']
# 遍历每个项目和对应的子图
for i, (ax, project) in enumerate(zip(axs.ravel(), SumTrendsList)):
    # 确保data和angles长度一致
    data = np.concatenate((project, [project[0]]))
    angles = np.concatenate((angles[:6], [angles[0]]))

    # 在极坐标中绘制数据
    ax.plot(angles, data, color=colorList[i], marker='o')

    ax.set_title(nameList[i])  # 设置子图标题

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.show()
NewlyAddedAttention = pd.read_excel('Q4Data.xlsx')
NewlyAddedAttention.columns = ['Project', 'Popularity and Accessibility', 'Gender Equity', 'Sustainability',
                               'Inclusivity', 'Relevance and Innovation', 'Safety and Fair Play']

NewlyAdded=NewlyAddedAttention.iloc[:,1:7].values.tolist()
# 假设这是一些SDE的数据
sde_data = {
    "surfing":NewlyAdded[0],
    "cricket":NewlyAdded[1],
    "steeplechase":NewlyAdded[2],
    "e-sports":NewlyAdded[3],
    "Australianfootball":NewlyAdded[4],
    "Dragonboatrace": NewlyAdded[5]
}

# 评估每个SDE
for sde, scores in sde_data.items():
    sde_score = np.dot(final_weights, scores)
    print(f"{sde}: Score = {sde_score}")

# 将Score排序并可视化
sorted_sde_data = {k: np.dot(final_weights, v) for k, v in sde_data.items()}
sorted_sde_data = dict(sorted(sorted_sde_data.items(), key=lambda x: x[1], reverse=True)
)
plt.figure(figsize=(30, 10))
# 用不同的颜色
plt.bar(sorted_sde_data.keys(), sorted_sde_data.values(), color=['red', 'blue', 'green', 'purple', 'orange', 'yellow'], alpha=0.5)
plt.xticks(rotation=45)
plt.xlabel("SDE")
plt.ylabel("Score")
plt.title("SDE Scores")
plt.show()

# 构建判断矩阵（这里需要专家打分或者数据来填充）
judgment_matrix1 = np.array([
    [1,     1/0.8,  1/0.5,  1/0.6,  1/0.7,  1/0.85],
    [0.8,       1,  1/0.6,  1/0.8,  1/0.7,      1],
    [0.5,     0.6,      1,  1/1.2,  1/1.8,    1/2],
    [0.6,     0.8,    1.2,      1,  1/1.2,  1/1.3],
    [0.7,     0.7,    1.8,    1.2,      1,  1/1.1],
    [0.85,       1,      2,    1.3,    1.1,      1]
])

judgment_matrix2 = np.array([
    [1,     1/0.5,  1/0.4,  1/0.6,  1/0.7,  1/0.8],
    [0.5,       1,  1/0.5,  1/0.8,  1/0.9,      1],
    [0.4,     0.5,      1,  1/1.6,  1/1.5,    1/2],
    [0.6,     0.8,    1.6,      1,  1/1.2,  1/1.2],
    [0.7,     0.9,    1.5,    1.2,      1,  1/1.2],
    [0.8,       1,      2,    1.2,    1.2,      1]
])

judgment_matrix3 = np.array([
    [1,     1/0.8,  1/0.6,  1/0.9,  1/0.7,  1/0.8],
    [0.8,       1,  1/0.5,  1/0.8,  1/0.9,      1],
    [0.6,     0.5,      1,  1/1.2,  1/1.5,    1/2],
    [0.9,     0.8,    1.2,      1,  1/1.2,  1/1.8],
    [0.7,     0.9,    1.5,    1.2,      1,  1/1.2],
    [0.8,       1,      2,    1.8,    1.2,      1]
])

# 将三个判断矩阵作为子图显示
fig, axs = plt.subplots(1, 3, figsize=(30, 8))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(judgment_matrix1, annot=True, xticklabels=criteria, yticklabels=criteria, cmap=cmap, ax=axs[0])
sns.heatmap(judgment_matrix2, annot=True, xticklabels=criteria, yticklabels=criteria, cmap=cmap, ax=axs[1])
sns.heatmap(judgment_matrix3, annot=True, xticklabels=criteria, yticklabels=criteria, cmap=cmap, ax=axs[2])
# 不显示横坐标
for ax in axs:
    ax.set_xticks([])
plt.show()

# 计算权重
eigenvalues1, eigenvectors1 = eig(judgment_matrix1)
max_index1 = np.argmax(eigenvalues1)
max_eigenvalue1 = eigenvalues1[max_index1].real
weights_normalized1 = np.abs(eigenvectors1[:, max_index1].real)
final_weights1 = weights_normalized1 / weights_normalized1.sum()

eigenvalues2, eigenvectors2 = eig(judgment_matrix2)
max_index2 = np.argmax(eigenvalues2)
max_eigenvalue2 = eigenvalues2[max_index2].real
weights_normalized2 = np.abs(eigenvectors2[:, max_index2].real)
final_weights2 = weights_normalized2 / weights_normalized2.sum()

eigenvalues3, eigenvectors3 = eig(judgment_matrix3)
max_index3 = np.argmax(eigenvalues3)
max_eigenvalue3 = eigenvalues3[max_index3].real
weights_normalized3 = np.abs(eigenvectors3[:, max_index3].real)
final_weights3 = weights_normalized3 / weights_normalized3.sum()

# 一致性检验
CI1 = (max_eigenvalue1 - len(criteria)) / (len(criteria) - 1)
RI1 = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32]  # 随机一致性指数，取决于矩阵的阶数
CR1 = CI1 / RI1[len(criteria) - 1]
# print('一致性检验1:', CR1)

CI2 = (max_eigenvalue2 - len(criteria)) / (len(criteria) - 1)
RI2 = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32]  # 随机一致性指数，取决于矩阵的阶数
CR2 = CI2 / RI2[len(criteria) - 1]
# print('一致性检验2:', CR2)

CI3 = (max_eigenvalue3 - len(criteria)) / (len(criteria) - 1)
RI3 = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32]  # 随机一致性指数，取决于矩阵的阶数
CR3 = CI3 / RI3[len(criteria) - 1]
# print('一致性检验3:', CR3)

# 序号和Consistency check存入dataframe
ConsistencyCheck = pd.DataFrame({ '序号': ['1', '2', '3'], 'Consistency check': [CR1, CR2, CR3]})
ConsistencyCheck.to_csv('Temp.csv', index=False)

SumTrends=AttentionQuery.iloc[55:61, 0:7]
SumTrendsList=SumTrends.iloc[:,1:7].values.tolist()

# 假设这是一些SDE的数据
sde_data1 = {
    "Baseball":SumTrendsList[0],
    "Softball":SumTrendsList[1],
    "Karate":SumTrendsList[2],
    "Swimming":SumTrendsList[3],
    "Basketball": SumTrendsList[4],
    "Jumping": SumTrendsList[5]
}

# 评估每个SDE
for sde, scores in sde_data1.items():
    sde_score = np.dot(final_weights1, scores)
    # print(f"{sde}: Score1 = {sde_score}")

# 将Score排序并可视化
sorted_sde_data1 = {k: np.dot(final_weights1, v) for k, v in sde_data1.items()}
sorted_sde_data1 = dict(sorted(sorted_sde_data1.items(), key=lambda x: x[1], reverse=True)
)

# 写入DataFrame
sortdf1 = pd.DataFrame(sorted_sde_data1.items(), columns=['SDE', 'Score1'])

# 假设这是一些SDE的数据
sde_data2 = {
    "Baseball":SumTrendsList[0],
    "Softball":SumTrendsList[1],
    "Karate":SumTrendsList[2],
    "Swimming":SumTrendsList[3],
    "Basketball": SumTrendsList[4],
    "Jumping": SumTrendsList[5]
}

# 评估每个SDE
for sde, scores in sde_data2.items():
    sde_score = np.dot(final_weights2, scores)
    # print(f"{sde}: Score2 = {sde_score}")

# 将Score排序并可视化
sorted_sde_data2 = {k: np.dot(final_weights2, v) for k, v in sde_data2.items()}
sorted_sde_data2 = dict(sorted(sorted_sde_data2.items(), key=lambda x: x[1], reverse=True)
)

sortdf2 = pd.DataFrame(sorted_sde_data2.items(), columns=['SDE', 'Score2'])

# 假设这是一些SDE的数据
sde_data3 = {
    "Baseball":SumTrendsList[0],
    "Softball":SumTrendsList[1],
    "Karate":SumTrendsList[2],
    "Swimming":SumTrendsList[3],
    "Basketball": SumTrendsList[4],
    "Jumping": SumTrendsList[5]
}

# 评估每个SDE
for sde, scores in sde_data3.items():
    sde_score = np.dot(final_weights3, scores)
    # print(f"{sde}: Score3 = {sde_score}")

# 将Score排序并可视化
sorted_sde_data3 = {k: np.dot(final_weights3, v) for k, v in sde_data3.items()}
sorted_sde_data3 = dict(sorted(sorted_sde_data3.items(), key=lambda x: x[1], reverse=True)
)

sortdf3 = pd.DataFrame(sorted_sde_data3.items(), columns=['SDE', 'Score3'])

# 将三幅图作为子图显示

fig, axs = plt.subplots(1, 3, figsize=(30, 8))

axs[0].bar(sorted_sde_data1.keys(), sorted_sde_data1.values(), color=['red', 'blue', 'green', 'purple', 'orange', 'yellow'], alpha=0.5)
axs[0].set_title("SDE Scores1")
axs[0].set_xlabel("SDE")
axs[0].set_ylabel("Score")
axs[0].tick_params(axis='x', rotation=45)

axs[1].bar(sorted_sde_data2.keys(), sorted_sde_data2.values(), color=['red', 'blue', 'green', 'purple', 'orange', 'yellow'], alpha=0.5)
axs[1].set_title("SDE Scores2")
axs[1].set_xlabel("SDE")
axs[1].set_ylabel("Score")
axs[1].tick_params(axis='x', rotation=45)

axs[2].bar(sorted_sde_data3.keys(), sorted_sde_data3.values(), color=['red', 'blue', 'green', 'purple', 'orange', 'yellow'], alpha=0.5)
axs[2].set_title("SDE Scores3")
axs[2].set_xlabel("SDE")
axs[2].set_ylabel("Score")
axs[2].tick_params(axis='x', rotation=45)


sortdf1 = sortdf1.set_index('SDE')
sortdf2 = sortdf2.set_index('SDE')
sortdf3 = sortdf3.set_index('SDE')

sortdf = pd.concat([sortdf1, sortdf2, sortdf3], axis=1)
sortdf.columns = ['Score1', 'Score2', 'Score3']
print(sortdf)