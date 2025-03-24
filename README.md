# COMP7409_Group34_2025
### 2025 第一学期 COMP7409_Group34 的小组期末项目 组员(孙昱轩，田潇文(组长)，王鹏一，吴佳吟，张彤彤)(按拼音排序)
1.后续的项目规划和分工会写在此readme file中.
2.各位更新的话记得同步更新此readme file且在commit changes时写好comment便于其他组员查看进度。
3.如果有问题以及建议请及时在微信群中反馈。

### 目前项目为: Forecasting Short Term Returns in Popular Cryptocurrencies Based on Machine Learning

### 目标为: __利用机器学习方法预测流行加密货币的短期回报__

### 项目模块分为:
1. 数据收集与清洗模块
2. 模型1训练与评估(catboost)
3. 模型2训练与评估(LightGBM)
4. 模型集成与结果分析
5. UI模块与各模块的整合

### 准备使用的模型数据为:
1.[https://www.kaggle.com/competitions/g-research-crypto-forecasting](https://www.kaggle.com/competitions/g-research-crypto-forecasting/data)

2025.3.21 项目分工
### 1.数据收集与清洗模块：(孙昱轩)
1. 数据收集与预处理：
收集14种加密货币的真实交易数据，包括最高价、最低价、开盘价、收盘价、成交量等; 
对数据进行初步清洗，处理缺失值和异常值;
2. 特征提取与大盘指数构建
提取最近N日的涨幅、量比等特征;
构建包括14种货币的大盘指数，并计算涨幅、量比等特征。
3. 特征工程与数据准备：
对提取的特征进行进一步处理，如归一化、标准化等。
将数据分为训练集和测试集，确保数据适合用于机器学习模型。
### 2.模型训练与评估(catboost & LightGBM)：(吴佳吟 & 张彤彤)
使用LightGBM和CatBoost模型进行训练;
使用RMSE（均方根误差）作为损失函数，评估模型性能。
### 3.模型集成与结果分析：（王鹏一）
分析集成模型的性能，比较单一模型和集成模型的效果；
尝试别的模型对比，如RNN,transformer,conv1D；加入市场情感分析（自由发挥，不一定要真敲代码）
### 4.UI模块与各模块的整合：（田潇文）
此项目UI及代码模块化整合。

项目进度
2025.03.24
增加了获取数据集的函数
