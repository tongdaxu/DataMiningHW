# %%
import os
import sys
import time
import numpy as np
import pandas as pd
# import xgboost as xgb
from sklearn.metrics import roc_auc_score
# from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# %% 1. Data Preparation
data = pd.read_csv('/data2/mengfanjin/meituan/data_wajue/HousingData.csv')

labels = data['MEDV']

# features = merged.drop(['global_id', 'label', 'userid', 'itemid'], axis=1)
# features = data.drop(['global_id', 'label', 'userid', 'itemid', 'timestamp'], axis=1)
features = data
features = features.drop(['MEDV'], axis=1)



# # %% 1. Data online
# data = pd.read_csv('/data2/mengfanjin/meituan/data_wajue/OnlineNewsPopularity.csv')

# # print('test_data:', data.columns)
# labels = data[' shares']
# features = data
# features = features.drop(['url',' shares'], axis=1)


# #wine
# data = pd.read_csv('/data2/mengfanjin/meituan/data_wajue/winequality-red.csv',sep=';')

# print('train_data:', data.columns)
# labels = data['quality']

# # features = merged.drop(['global_id', 'label', 'userid', 'itemid'], axis=1)
# # features = data.drop(['global_id', 'label', 'userid', 'itemid', 'timestamp'], axis=1)
# features = data
# features = features.drop(['quality'], axis=1)



columns_with_na = features.columns[features.isna().any()].tolist()

# 使用均值填充空缺值
for column in columns_with_na:
    column_mean = features[column].mean()
    features[column].fillna(column_mean, inplace=True)


# features.fillna(-999, inplace=True)
assert features.isnull().sum().sum() == 0


def get_result(random_seed=0):

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=random_seed)

    train_features = X_train
    train_labels = y_train
    valid_features = X_test
    valid_labels = y_test
    # %% 2. Model Training
    # train_size = int(len(features) * 0.8)
    # train_idx = np.random.choice(len(features), train_size, replace=False)
    # valid_idx = np.setdiff1d(list(range(len(features))), train_idx)
    # train_features = features.iloc[train_idx]
    # train_labels = labels.iloc[train_idx]
    # valid_features = features.iloc[valid_idx]
    # valid_labels = labels.iloc[valid_idx]

    # 创建训练数据集
    train_data = lgb.Dataset(train_features, label=train_labels)

    # 创建验证数据集
    valid_data = lgb.Dataset(valid_features, label=valid_labels, reference=train_data)


        # # 定义超参数dict-white
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'max_depth':5,# 5,
    #     'num_leaves':7,# 8,
    #     'learning_rate':0.1,# 0.03,
    #     'feature_fraction': 0.9,#
    #     'verbose': -1,
    #     'reg_lambda':1,# 0.1,
    #     'num_iterations' : 10000,
    #     'return_best_model':True,
    #     'metric': 'rmse',  # 评估指标为AUC
    #     'bagging_fraction' : 0.8
    #     # 'early_stopping_round': 10, # 早停参数
    # }

    # red
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'max_depth':5,# 5,
    #     'num_leaves':8,# 8,
    #     'learning_rate':0.03,# 0.03,
    #     'feature_fraction': 0.9,#
    #     'verbose': -1,
    #     'reg_lambda':0.1,# 0.1,
    #     'num_iterations' : 10000,
    #     'return_best_model':True,
    #     'metric': 'rmse',  # 评估指标为AUC
    #     'bagging_fraction' : 0.8
    #     # 'early_stopping_round': 10, # 早停参数
    # }

    #online
#     params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'max_depth': 3,
#     'num_leaves': 8,
#     'learning_rate': 0.01,
#     'feature_fraction': 0.9,#
#     'verbose': -1,
#     'reg_lambda': 1,
#     # 'lambda_l1' :10,
#     'num_iterations' : 10000,
#     'return_best_model':True,
#     'metric': 'rmse',  # 评估指标为AUC
#     'bagging_fraction' : 0.8
#     # 'early_stopping_round': 10, # 早停参数
# }
#     # mean_r2 0.02075282682626911 mean_mae 3029.2812592575306 mean_rmse 11603.111426045

    #house
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'max_depth':4,# 5,
        'num_leaves':5,# 8,
        'learning_rate':0.05,# 0.03,
        'feature_fraction': 0.9,#
        'verbose': -1,
        'reg_lambda':0,# 0.1,
        'num_iterations' : 10000,
        'return_best_model':True,
        'metric': 'rmse',  # 评估指标为AUC
        'bagging_fraction' : 0.8
        # 'early_stopping_round': 10, # 早停参数
    }


    callback=[lgb.early_stopping(stopping_rounds=100,verbose=True),
              lgb.log_evaluation(period=50,show_stdv=True)]


    # 训练模型
    clf = lgb.train(params, train_data, valid_sets=[valid_data],callbacks=callback)

    # 预测验证集
    valid_probs = clf.predict(valid_features, num_iteration=clf.best_iteration)


    # 计算 R2
    r2 = r2_score( valid_labels,valid_probs)

    # 计算 MAE
    mae = mean_absolute_error( valid_labels,valid_probs)

    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error( valid_labels,valid_probs))

    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    return r2,mae,rmse


r2_tot = []
mae_tot = []
rmse_tot = []

for seed in range(5):
    r2,mae,rmse = get_result(seed)
    r2_tot.append(r2)
    mae_tot.append(mae)
    rmse_tot.append(rmse)

# 计算列表的均值
mean_r2 = sum(r2_tot) / len(r2_tot)
mean_mae = sum(mae_tot) / len(mae_tot)
mean_rmse = sum(rmse_tot) / len(rmse_tot)

print(r2_tot,mae_tot,rmse_tot)
# 输出均值
print('mean_r2',mean_r2,'mean_mae',mean_mae,'mean_rmse',mean_rmse)

#2
# mean_r2 0.40784221989763125 mean_mae 0.4546895822203063 mean_rmse 0.5930700006395588

# # 计算均方根误差
# mse = mean_squared_error(valid_labels, valid_probs)
# rmse = np.sqrt(mse)
# print("Root Mean Squared Error:", rmse)

# r2 = r2_score(y_true, y_pred)
# print("R2 score:", r2)

# # 计算MAE
# mae = mean_absolute_error(y_true, y_pred)
# print("MAE:", mae)

# # 计算RMSE
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# print("RMSE:", rmse)







#0.67-0.97


## 设置参数
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'feature_fraction': 0.9,#
#     'verbose': -1,
#     'num_iterations' : 10000,
#     'return_best_model':True,
#     'metric': 'rmse',  # 评估指标为AUC
#     'bagging_fraction' : 0.8,
#     'early_stopping_rounds':50,
# }

# # 设置参数搜索空间
# param_grid = {
#     'num_leaves': [3,4,5,7,8],
#     'learning_rate': [0.1, 0.05, 0.03,0.01],
#     'max_depth': [3,4, 5],
#     'lambda_l1': [0, 0.1, 0.2, 0.5,1],

# }

# Best Parameters: {'lambda_l1': 1, 'learning_rate': 0.1, 'max_depth': 5, 'num_leaves': 7}
# gbm = lgb.LGBMRegressor(
#                     n_estimators=10000, # 使用多少个弱分类器
#                     objective='regression',
#                     boosting_type = 'gbdt',
#                     feature_fraction = 0.9,
#                     verbose =-1,
#                     metric = 'rmse',
#                     bagging_fraction = 0.8,
#                     early_stopping_rounds = 50

#                 )

# # 使用GridSearchCV进行参数搜索
# grid_search = GridSearchCV(estimator=gbm,
#                            param_grid=param_grid,
#                            scoring='neg_mean_squared_error',
#                            cv=5,
#                            verbose=1)

# # 训练模型并搜索最佳参数
# grid_search.fit(train_features, train_labels, eval_set=[(valid_features, valid_labels)])

# # 输出最佳参数和对应的模型性能
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", np.sqrt(-grid_search.best_score_))







