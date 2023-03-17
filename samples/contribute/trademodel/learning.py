DATASET_ALPHA158_CLASS = "Alpha158"
CSI300_MARKET = 'all'
import singletrader
from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import Alpha158
from singletrader.shared.utility import save_pkl,load_pkl,load_pkls
from configs import basic_config, GBDT_MODEL
Alpha158.parse_config_to_fields = basic_config
from utils import get_period_split
from qlib.data.dataset import DatasetH
from samples.contribute.data.models import LinearModel
import pandas as pd

start = '2010-01-01'
end = '2023-01-31'


LINREG = {
    "class": LinearModel,
    "kwargs": {
    "estimator":"ols",
    "fit_intercept":True,
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




if __name__ == '__main__':
    train,valid,test = get_period_split()
    compare_result = {}
    pred_lins = []
    pred_lgbs = []
    for i in range(len(train)):

        model_lin = init_instance_by_config(LINREG)
        model_lgb = init_instance_by_config(GBDT_MODEL)
        dataset = init_instance_by_config(get_dataset_config(
                                                                train=train[i],
                                                                valid=valid[i],
                                                                test=test[i]
                                                             ))
        date_year = test[i][0]
        model_lin.fit(dataset)
        model_lgb.fit(dataset)
        pred_lin = model_lin.predict(dataset)
        pred_lgb = model_lgb.predict(dataset)

        pred_lins.append(pred_lin)
        pred_lgbs.append(pred_lgb)        


        pr_lin = pd.concat([pred_lin, dataset.prepare('test',col_set=['label'])],axis=1)
        pr_lgb = pd.concat([pred_lgb, dataset.prepare('test',col_set=['label'])],axis=1)



        result_lin = pr_lin.groupby(level=0).apply(lambda x:x.corr().iloc[0,1]).describe()
        result_lgb = pr_lgb.groupby(level=0).apply(lambda x:x.corr().iloc[0,1]).describe()
        
        
        compare_result[test[i][:10]] =pd.concat([result_lin,result_lgb],axis=1)
        print("===")

    
    model = init_instance_by_config(GBDT_MODEL)
    dataset = init_instance_by_config(get_dataset_config())
    
    
    from singletrader.factortesting.factor_testing import FactorEvaluation
    from singletrader.datasdk.qlibapi.constructor.base import MultiFactor
    import pandas as pd
    from singletrader.performance.common import performance_indicator
    
    fields = ['$close','$open','$high','$low','$avg','$volume']
    bars = ['close','open','high','low','avg','volume']
    names = bars
    
    mf = MultiFactor(field=fields,name=names,start_date='2021-01-01',end_date='2023-03-31')
    data=mf._data
    
    dataset = init_instance_by_config(get_dataset_config())
    from qlib.data.dataset import DatasetH

    model = init_instance_by_config(GBDT_MODEL)
    model.fit(dataset)

    pred = model.predict(dataset)
    pred_true = pd.concat([pred,dataset.prepare('test',col_set=['label'])],axis=1)

    fe = FactorEvaluation(bar_data=data.swaplevel(0,1),factor_data=pred)
    gp = fe.get_group_returns(base='avg',groups=10,holding_period=1)
    
    # # model = init_instance_by_config(CSI300_GBDT_TASK["model"])
    print(r'===')