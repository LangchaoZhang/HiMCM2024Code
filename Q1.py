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

TPDS=pd.read_csv('1-1.TPDS.csv')
# 去除全为缺失值的行
TPDS=TPDS.dropna(how='all')
# 年份列转换为字符串
TPDS['年份']=TPDS['年份'].astype(str)

plt.figure(figsize=(30,10))
for i in range(1, len(TPDS.columns)):
    plt.plot(TPDS['年份'], TPDS[TPDS.columns[i]], label=TPDS.columns[i], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of people')
plt.legend()

# TPDS按照Basketball进行排序
TPDS=TPDS.sort_values(by='Athletics', ascending=False)
TPDS.head()

TMDS=pd.read_csv('1-2.TMDS.csv')
# 去除全为缺失值的行
TMDS=TMDS.dropna(how='all')
# 年份列转换为字符串
TMDS['年份']=TMDS['年份'].astype(str)

plt.figure(figsize=(30,10))
for i in range(1, len(TMDS.columns)):
    plt.plot(TMDS['年份'], TMDS[TMDS.columns[i]], label=TMDS.columns[i], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of Medals')
plt.legend()
plt.show()


# 利用PGDS,PPDS,PTDS来绘制堆叠柱状图，其中PGDS是金牌占比，PPDS是银牌占比，PTDS是铜牌占比
PGDS=pd.read_csv('1-3.PGDS.csv')
PSDS=pd.read_csv('1-4.PSDS.csv')
PBDS=pd.read_csv('1-5.PBDS.csv')

# 年份列转换为字符串
PGDS['年份']=PGDS['年份'].astype(str)
PSDS['年份']=PSDS['年份'].astype(str)
PBDS['年份']=PBDS['年份'].astype(str)

# 生成堆叠柱状图
fig, ax = plt.subplots(figsize=(30,10))
bar_width = 0.25
bar_l = range(len(PGDS['年份']))
bar_l1 = [i+bar_width for i in bar_l]
bar_l2 = [i+bar_width*2 for i in bar_l]
bar_l3 = [i+bar_width*3 for i in bar_l]

plt.bar(bar_l1, PGDS['Athletics'], width=bar_width, label='Gold medal ratio',color='pink')
plt.bar(bar_l2, PSDS['Athletics'], width=bar_width, label='Silver medal ratio',color='silver')
plt.bar(bar_l3, PBDS['Athletics'], width=bar_width, label='Bronze medal ratio',color='skyblue')

plt.xlabel('Year')
plt.ylabel('Medal Ratio')

plt.xticks(bar_l1, PGDS['年份'])

plt.legend()
plt.show()

# 数据处理流程
GENDER=pd.read_csv('2-Gender.csv')
# 去除全为缺失值的行
GENDER.dropna(how='all')


# 年份列转换为字符串
# GENDER['年份']= GENDER['年份'].astype(str)

plt.figure(figsize=(30,10))
for i in range(1,len(GENDER.columns)):
    plt.plot(GENDER['年份'],GENDER[GENDER.columns[i]],label=GENDER.columns[i], marker='o')

plt.xlabel('Year')
plt.ylabel('Ratio')
plt.legend()
plt.show()

Environment=pd.read_csv('3-1.Environment.csv')
Economic=pd.read_csv('3-2.Economic.csv')
Public=pd.read_csv('3-3.Public.csv')
Area=pd.read_csv('3-4.Area.csv')


Environment.dropna(how='all')
Environment['年份']= Environment['年份'].astype(str)
Environment=Environment[Environment['年份']>'1984']

Economic.dropna(how='all')
Economic['年份']= Economic['年份'].astype(str)
Economic=Economic[Economic['年份']>'1984']

Public.dropna(how='all')
Public['年份']= Public['年份'].astype(str)
Public=Public[Public['年份']>'1984']

Area.dropna(how='all')
Area['年份']= Area['年份'].astype(str)
Area=Area[Area['年份']>'1984']

fig, ax = plt.subplots(2, 2, figsize=(30,15))
for i in range(1, len(Environment.columns)):
    ax[0, 0].plot(Environment['年份'], Environment[Environment.columns[i]], label=Environment.columns[i], marker='o')
    ax[0, 0].set_xlabel('Year')
    ax[0, 0].set_ylabel('Environment index')
    # ax[0, 0].legend()

for i in range(1, len(Economic.columns)):
    ax[0, 1].plot(Economic['年份'], Economic[Economic.columns[i]], label=Economic.columns[i], marker='o')
    ax[0, 1].set_xlabel('Year')
    ax[0, 1].set_ylabel('Economic index')
    # ax[0, 1].legend()

for i in range(1, len(Public.columns)):
    ax[1, 0].plot(Public['年份'], Public[Public.columns[i]], label=Public.columns[i], marker='o')
    ax[1, 0].set_xlabel('Year')
    ax[1, 0].set_ylabel('Public index')
    # ax[1, 0].legend()

for i in range(1, len(Area.columns)):
    ax[1, 1].plot(Area['年份'], Area[Area.columns[i]], label=Area.columns[i], marker='o')
    ax[1, 1].set_xlabel('Year')
    ax[1, 1].set_ylabel('Area')
    # ax[1, 1].legend()

plt.show()


Inclusive=pd.read_csv('4-Inclusive.csv')

Inclusive.dropna(how='all')
# 年份列转换为字符串
Inclusive['年份']= Inclusive['年份'].astype(str)

plt.figure(figsize=(30,10))
for i in range(1,len(Inclusive.columns)):
    plt.plot(Inclusive['年份'],Inclusive[Inclusive.columns[i]],label=Inclusive.columns[i], marker='o')

plt.xlabel('Year')
plt.ylabel('Number of Countries')
plt.legend()
plt.show()

AO20=pd.read_csv('5-1.AO20.csv')
AO23=pd.read_csv('5-5.AO23.csv')
AO34=pd.read_csv('5-6.AO34.csv')

# 年份列转换为字符串
AO20['年份']=AO20['年份'].astype(str)
AO23['年份']=AO23['年份'].astype(str)
AO34['年份']=AO34['年份'].astype(str)

# 生成堆叠柱状图
fig, ax = plt.subplots(figsize=(30,10))
bar_width = 0.25
bar_l = range(len(AO20['年份']))
bar_l1 = [i+bar_width for i in bar_l]
bar_l2 = [i+bar_width*2 for i in bar_l]
bar_l3 = [i+bar_width*3 for i in bar_l]

plt.bar(bar_l1, AO20['Athletics'], width=bar_width, label='Below 20 ratio',color='pink')
plt.bar(bar_l2, AO23['Athletics'], width=bar_width, label='Range 20-30 ratio',color='silver')
plt.bar(bar_l3, AO34['Athletics'], width=bar_width, label='Range 30-40 ratio',color='skyblue')

plt.xlabel('Year')
plt.ylabel('Age Range')
plt.legend()
plt.show()


import numpy as np

AO20=pd.read_csv('5-1.AO20.csv')
AO40=pd.read_csv('5-3.AO40.csv')


# 年份列转换为字符串
AO20['年份']=AO20['年份'].astype(str)
AO40['年份']=AO40['年份'].astype(str)


AVAR=pd.read_csv('5-4.AVAR.csv')
AVAR.dropna(how='all')

AVARproject=AVAR.columns[1:16]


fig, ax = plt.subplots(2, 2, figsize=(30,10))

pro = AVARproject[0]
alldata = np.array([
    AO20[pro],
    AO40[pro],
    AVAR[pro]
])
avg = np.mean(alldata, axis=0)
std = np.std(alldata, axis=0)
ax[0, 0].plot(AO20['年份'], avg, label=f"{pro}", linewidth=3.0,c='red')
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
ax[0, 0].fill_between(AO20['年份'], r1, r2, color='skyblue', alpha=0.7)
ax[0, 0].legend(loc='upper left')
ax[0, 0].set_xlabel('Year')
ax[0, 0].set_ylabel('Value')

pro = AVARproject[2]
alldata = np.array([
    AO20[pro],
    AO40[pro],
    AVAR[pro]
])
avg = np.mean(alldata, axis=0)
std = np.std(alldata, axis=0)
ax[0, 1].plot(AO20['年份'], avg, label=f"{pro}", linewidth=3.0,c='red')
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
ax[0, 1].fill_between(AO20['年份'], r1, r2, color='pink', alpha=0.8)
ax[0, 1].legend(loc='upper left')
ax[0, 1].set_xlabel('Year')

pro = AVARproject[12]
alldata = np.array([
    AO20[pro],
    AO40[pro],
    AVAR[pro]
])
avg = np.mean(alldata, axis=0)
std = np.std(alldata, axis=0)
ax[1, 0].plot(AO20['年份'], avg, label=f"{pro}", linewidth=3.0,c='red')
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
ax[1, 0].fill_between(AO20['年份'], r1, r2, color='b', alpha=0.2)
ax[1, 0].legend(loc='upper left')
ax[1, 0].set_xlabel('Year')
ax[1, 0].set_ylabel('Value')

pro = AVARproject[3]
alldata = np.array([
    AO20[pro],
    AO40[pro],
    AVAR[pro]
])
avg = np.mean(alldata, axis=0)
std = np.std(alldata, axis=0)
ax[1, 1].plot(AO20['年份'], avg, label=f"{pro}", linewidth=3.0,c='red')
r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
ax[1, 1].fill_between(AO20['年份'], r1, r2, color='g', alpha=0.2)
ax[1, 1].legend(loc='upper left')
ax[1, 1].set_xlabel('Year')
plt.show()


Indicators=pd.read_csv('6-1.Indicators.csv')

Indicators.dropna(how='all')
row_sums = Indicators.sum(axis=1)-Indicators['年份']
Indicators['年份']= Indicators['年份'].astype(str)
plt.figure(figsize=(30,10))
for i in range(1,len(Indicators.columns)):
    plt.plot(Indicators['年份'],1-Indicators[Indicators.columns[i]]/row_sums,label=Indicators.columns[i], marker='o')

plt.xlabel('Year')
plt.ylabel('Safety Score')
plt.legend()
plt.show()
