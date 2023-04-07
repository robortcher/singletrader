DATASET_ALPHA158_CLASS = "Alpha158"
CSI300_MARKET = 'all'
import singletrader
from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import Alpha158
from singletrader.shared.utility import save_pkl,load_pkl,load_pkls
from config import get_feature_configs
Alpha158.parse_config_to_fields = get_feature_configs
from singletrader.shared.utility import get_period_split
from singletrader.shared.utility import check_and_mkdir
from singletrader.datasdk.sql.dataapi import get_index_cons

start = '2005-01-01'
end = '2023-01-31'



GBDT_MODEL = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8879,
        "learning_rate": 0.0421,
        "subsample": 0.8789,
        "lambda_l1": 205.6999,
        "lambda_l2": 580.9768,
        "max_depth": 8,
        "num_leaves": 210,
        "num_threads": 20,
        "early_stopping_rounds":200,
        "min_data_per_group":1000,
    },
}

def get_data_handler_config(
    start_time="2010-01-01",
    end_time="2023-03-31",
    fit_start_time="<dataset.kwargs.segments.train.0>",
    fit_end_time="<dataset.kwargs.segments.train.1>",
    instruments=CSI300_MARKET,
):
    return {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "instruments": instruments,
    }

def get_dataset_config(
    dataset_class=Alpha158,
    train=("2010-01-01", "2019-12-31"),
    valid=("2020-01-01", "2021-12-31"),
    test=("2022-01-01", "2023-03-31"),
    handler_kwargs={"instruments": CSI300_MARKET},
):
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": dataset_class,
                # "module_path": "qlib.contrib.data.handler",
                "kwargs": get_data_handler_config(**handler_kwargs),
            },
            "segments": {
                "train": train,
                "valid": valid,
                "test": test,
            },
        },
    }



def main(X_train,X_test,y_train,y_test):
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error

    # 创建随机回归数据集
    # X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, noise=0.5, random_state=42)

    # 将数据集拆分为训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建AdaBoost回归器
    base_estimator = DecisionTreeRegressor(max_depth=1)
    boosting_reg = AdaBoostRegressor(base_estimator=base_estimator, random_state=42)

    # 设置参数搜索范围
    param_grid = {'n_estimators': [50, 100, 200],
                'learning_rate': [0.1, 0.3, 0.5]}

    # 使用网格搜索来选择最佳参数组合
    grid_search = GridSearchCV(boosting_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # 打印最佳参数组合和测试集均方误差
    print("Best parameters: ", grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Test MSE: {:.2f}".format(mse))



if __name__ == '__main__':
    train,valid,test = get_period_split(fix_train_year=None)
    # index_cons = get_index_cons(start_date='2005-10-01')
    # index_cons.index = index_cons.index.set_names(['datetime','instruement'])

    for i in range(len(train)):
        
        model = init_instance_by_config(GBDT_MODEL)
        dataset = init_instance_by_config(get_dataset_config(handler_kwargs={"instruments": 'all'},
                                                                train=train[i],
                                                                valid=valid[i],
                                                                test=test[i]
                                                             ))
        
        # X_train, X_test, y_train, y_test = pd.concat(dataset.prepare(['train','test']))
        
        date_year = test[i][0]
        # dataset.handler._learn = dataset.handler._learn[index_cons['zz1000'].reindex(dataset.handler._learn.index)>0]
        model.fit(dataset)
        save_pkl(model, f'D:/projects/singletrader_pro/samples/factor_mining/tamplate01/model/{date_year}.pkl')
        pred = model.predict(dataset)
        save_pkl(pred, f'D:/projects/singletrader_pro/samples/factor_mining/tamplate01/predict/{date_year}.pkl')
    
    
    
    model = init_instance_by_config(GBDT_MODEL)
    dataset = init_instance_by_config(get_dataset_config())
    
    
