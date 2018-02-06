import pandas as pd
import numpy as np
import preprocessing as prep
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
import xgboost as xgb
import time
from datetime import datetime

xgb_params = {
    'eta': 0.06,
    'max_depth': 8,
    'subsample': 0.6,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}


print('read')
# macro = [
#     'usdrub', 'eurrub', 'brent',
#     'balance_trade', 'balance_trade_growth', 'micex_rgbi_tr', 'micex_cbi_tr',
#     'deposits_rate', 'mortgage_value', 'mortgage_rate']
data = pd.read_csv('hakaton-fin.csv',nrows=100000)
# macro_DF = pd.read_csv('macro.csv', parse_dates=['timestamp'], usecols=(['timestamp'] + macro))
data.columns = ["id",
                "good_id",
                "set_id",
                "timestamp",
                "device_id",
                "shop_id",
                "check_type",
                "total_cost",
                "total_cashback",
                "total_discount_pc",
                "total_discount",
                "total_tax_pc",
                "total_tax",
                "items_count",
                "item_name",
                "item_type",
                "um",
                "qnt",
                "price",
                "sum_price",
                "oper_discount",
                "result_sum",
                "purchase_price",
                "onhand_qnt",
                "region",
                "inn",
                "okved_full",
                "okved_description",
                "lat",
                "lng",
                "category_id",
                "category_name"]

data['lng'] = data['lng'].apply(lambda x: 0 if type(x)==str else x)
data['lat'] = data['lat'].apply(lambda x: 0 if type(x)==str else x)

train=data.drop(["okved_description","okved_full", "inn", "qnt","item_type",
                      "items_count","total_tax",  "total_discount","total_discount_pc","total_cashback",
                      "total_cost","check_type","shop_id","device_id","set_id","good_id","id","item_name","category_name",
                 "result_sum","sum_price","region"], axis=1)


# 47.11.3
# print('data')
# for d in data['item_name']:
#     print(d)
# exit()
# for col in data.columns:
#     # print(col)
#     for i in data[col]:
#         if i == '47.11.3':
#             print(col)
#             break
#     # prep.print_missing_statistics(data[col]);
# exit()
# print(data["timestamp"])


# print('Preprocessing -->> Macro Features:')
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["timestamp_day"] = train["timestamp"].apply(lambda d: d.day)
train["timestamp_month"]=train["timestamp"].apply(lambda d: d.month)
train["timestamp_week"] = train["timestamp"].apply(lambda d: d.weekday())
train = train.drop(["timestamp"],axis=1)

measure_to_cat = {v: i for i,v in enumerate(["","шт", "кг", "л", "м", "км", "м2", "м3", "компл", "упак", "ед", "дроб"])}

train["um"] = train["um"].apply(lambda x: measure_to_cat.get(x, 0))

y_train = train["oper_discount"]
train = train.drop(["oper_discount"],axis=1)


print(train.columns)

# train = np.array(train)
# train = train.astype(float)


dtrain = xgb.DMatrix(train, y_train)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1200, early_stopping_rounds=20, verbose_eval=20, show_stdv=False)
num_boost_rounds = len(cv_output)

print('fit')
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
model.save_model("procent_pizda"+".model")












