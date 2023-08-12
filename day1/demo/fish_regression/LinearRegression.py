# %%
# 读取数据
import pandas as pd

data = pd.read_csv('Fish.csv')
df = data.copy()
# 随机选取其中十个样本展示
df.sample(10)
# %%
# 查看数据信息
df.info()
# %%
# 查看数据是否有缺失
print('Is there any NaN in the dataset: {}'.format(df.isnull().values.any()))
# %%
# 相关系数矩阵
df.corr()
# %%
# 相关系数矩阵可视化
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='YlGnBu')
# %%
# correlation plot
sns.pairplot(df, kind='scatter', hue='Species')
# %%
# 描述数值型数据，其中`.T`表转置
df.describe().T
# %%
# 回归标签（预测目标）为鱼的重量
# 特征（自变量）为其他数值型特征
y = df['Weight']
X = df.iloc[:, 2:]
# %%
# 分离训练集与测试集并输出其shape
import numpy as np
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print('X_train: {}'.format(np.shape(X_train)))
print('y_train: {}'.format(np.shape(y_train)))
print('X_test: {}'.format(np.shape(X_test)))
print('y_test: {}'.format(np.shape(y_test)))
# %%
# 线性回归模型拟合
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
# %%
# 输出模型参数
print('Model intercept: ', reg.intercept_)
print('Model coefficients: ', reg.coef_)
# %%
# 输出模型
print('y = {} + {} * X1 + {} * X2 + {} * X3 + {} * X4 + {} * X5'.format(reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3], reg.coef_[4]))
# %%
# 评估模型
from sklearn.metrics import r2_score

r2_score(y_train, reg.predict(X_train))
# %%
# k-fold评估模型
from sklearn.model_selection import cross_val_score

cross_val_score(reg, X_train, y_train, cv=10, scoring='r2').mean()
# %%
# 预测及其评估
y_pred = reg.predict(X_test)
r2_score(y_test, y_pred)
# %%
