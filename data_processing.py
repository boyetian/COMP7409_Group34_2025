import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler    # RobustScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.simplefilter("ignore") # 忽略所有警告
pd.set_option("display.max_columns", None) # 设置 Pandas 显示选项，显示所有列
cut_type = 'peryear' # 定义切分类型为按年
train_type = 'scilearn' # 定义训练类型为 scikit-learn
data_type = 'all' # 定义数据类型为所有数据

####读取数据(根据具体路径)
df_asset_details = pd.read_csv("D:/HKU/COMP7409/project/solution/input/g-research-crypto-forecasting/asset_details.csv").sort_values("Asset_ID")
df = pd.read_csv("D:/HKU/COMP7409/project/solution/input/g-research-crypto-forecasting/train.csv")
df_sup = pd.read_csv("D:/HKU/COMP7409/project/solution/input/g-research-crypto-forecasting/supplemental_train.csv")
df_sup = df_sup[df_sup.timestamp>df.timestamp.max()]
df = pd.concat([df,df_sup]).reset_index(drop=True)

####数据预处理
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")  #将时间戳转换为datetime格式
df = df.replace([np.inf, -np.inf], np.nan)  #替换无穷大值为NaN
df = df.sort_values(by='datetime').reset_index(drop=True)  #按时间排序并重置索引
df = df[~df.Target.isna()].reset_index(drop=True)  #处理Target缺失值

####特征提取
time_windows = [15, 30, 60, 90, 150, 600, 1500]
for window in time_windows:
    df[f'Close_now_{window}'] = df.groupby('Asset_ID')['Close'].shift(window) / df['Close']  #计算涨幅
    df[f'Volume_now_{window}'] = df.groupby('Asset_ID')['Volume'].shift(window) / df['Volume']  #计算量比

####计算加权平均价格
def get_weighted_asset_feature(df, col):
    df['w'] = df['Asset_ID'].map(df_asset_details.set_index(keys='Asset_ID')['Weight'])
    weight_sum = df_asset_details.Weight.sum()

    df['W_'+col] = df.w * df[col]
    time_group = df.groupby('datetime')

    m = time_group['W_'+col].sum() / time_group['w'].sum()

    df.set_index(keys=['datetime'], inplace=True)
    df['W_'+col] = m
    df.reset_index(inplace=True)
    return df

df = get_weighted_asset_feature(df, 'Close_now_15')

####划分训练集与测试集(8：2)
split_date = "2021-06-13 00:00:00" 
df_train = df[df["datetime"] < split_date].reset_index(drop=True)
df_test = df[df["datetime"] >= split_date].reset_index(drop=True)


####释放内存
import gc
del df
gc.collect()
