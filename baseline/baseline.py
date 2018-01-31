import pandas as pd
import numpy as np

from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

train = pd.read_table('../data/train_20171215.txt', engine='python')
train.describe()

actions1 = train.groupby(['date', 'day_of_week'], as_index=False)['cnt'].agg({'count1': np.sum})

df_train_target = actions1['count1'].values
df_train_data = actions1.drop(['count1'], axis=1).values

# 切分数据（训练集和测试集）
cv = cross_validation.ShuffleSplit(len(df_train_data), n_iter=5, test_size=0.2, random_state=0)

print("GradientBoostingRegressor")
for train, test in cv:
    gbdt = GradientBoostingRegressor().fit(df_train_data[train], df_train_target[train])
    result1 = gbdt.predict(df_train_data[test])
    print(mean_squared_error(result1, df_train_target[test]))
    print('......')


