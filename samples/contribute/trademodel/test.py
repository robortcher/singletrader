DATASET_ALPHA158_CLASS = "Alpha158"
CSI300_MARKET = 'all'
import singletrader
from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import Alpha158
from singletrader.shared.utility import save_pkl,load_pkl,load_pkls
from configs import basic_config, GBDT_MODEL,basic_config2
# Alpha158.parse_config_to_fields = basic_config2
from utils import get_period_split
from singletrader.shared.utility import check_and_mkdir
from singletrader.datasdk.sql.dataapi import get_index_cons

start = '2005-01-01'
end = '2023-01-31'


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
    # index_cons = get_index_cons(start_date='2005-10-01')
    # index_cons.index = index_cons.index.set_names(['datetime','instruement'])

    for i in range(len(train)):
        
        model = init_instance_by_config(GBDT_MODEL)
        dataset = init_instance_by_config(get_dataset_config(handler_kwargs={"instruments": 'zz1800'},
                                                                train=train[i],
                                                                valid=valid[i],
                                                                test=test[i]
                                                             ))
        date_year = test[i][0]
        # dataset.handler._learn = dataset.handler._learn[index_cons['zz1000'].reindex(dataset.handler._learn.index)>0]
        model.fit(dataset)
        save_pkl(model, f'D:/projects/singletrader_pro/samples/contribute/trademodel/models500/{date_year}.pkl')
        pred = model.predict(dataset)
        save_pkl(pred, f'D:/projects/singletrader_pro/samples/contribute/trademodel/predict500/{date_year}.pkl')
    
    
    
    model = init_instance_by_config(GBDT_MODEL)
    dataset = init_instance_by_config(get_dataset_config())
    
    
