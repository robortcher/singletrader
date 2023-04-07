from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.utils import get_callable_kwargs
from qlib.data.dataset import processor as processor_module
from inspect import getfullargspec
from abc import abstractmethod
from qlib.utils import init_instance_by_config
import singletrader

def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l

def get_data_handler_config(
    start_time="2010-01-01",
    end_time="2023-03-31",
    fit_start_time="<dataset.kwargs.segments.train.0>",
    fit_end_time="<dataset.kwargs.segments.train.1>",
    instruments="all",
):
    return {
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "instruments": instruments,
    }

def get_dataset_config(
    dataset_class="Alpha158",
    train=("2015-01-01", "2019-12-31"),
    valid=("2020-01-01", "2021-12-31"),
    test=("2022-01-01", "2023-03-31"),
    handler_kwargs={"instruments": "all"},
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


class CustomDataHandler(DataHandlerLP):
    def __init__(
        self,
        instruments="all",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors= [{"class": "DropnaProcessor","kwargs": {"fields_group": "feature"}},
                {"class": "DropnaLabel"}],
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processor=None,
        **kwargs
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processor": inst_processor,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs
        )

    def get_feature_config(self):
        windows  = [1,2,5,10,20]
        fields=[]
        names =[]
        # n日涨幅
        fields += ["$close / Ref($close,%d) - 1" %d for d in windows]
        names += ["daily_return%d" %d for d in windows]

        # 收益波幅
        fields += ["($close / Ref($close,1) - 1) / Std($close / Ref($close,1) - 1, %d)" %d for d in (20,60,120)]
        names += ["retvol%d" % d for  d in (20,60,120)]

        # 换手率
        fields += ["Sum($turnover_ratio,%d)" %d for d in windows]
        names += ["TO%d" %d for d in windows]

        # 换手比
        fields += ["$turnover_ratio / Mean($turnover_ratio,%d)" %d for d in (20,60,120)]
        names += ["TOR%d" %d for d in (20,60,120)]
        
        
        return fields,names

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($avg, -1) - 1"], ["LABEL0"]



if __name__ == '__main__':
    from sklearn import tree
    import pandas as pd
    import numpy as np

    def xs_qcut(data, n=5, max_loss=0.2):
        """截面分组"""
        length = data.__len__()
        data = pd.DataFrame(data)
        def _qcut(data, n=n):
            try:
                res = pd.qcut(data,n,duplicates='drop',labels=list(range(n)))
            except:
                res =  pd.Series(np.nan,index = data.index,name=data.name)
            return res
        
        if type(data) is pd.DataFrame:
            
            res = data.dropna().groupby(level= 0).apply(lambda x:x.apply(lambda x:_qcut(x)))

            return res
        else:
            res = data.dropna().groupby(level= 0).apply(lambda x:x.apply(lambda x:_qcut(x)))
            return res

    dataset = init_instance_by_config(get_dataset_config(dataset_class=CustomDataHandler))
    data =  dataset.handler._learn.droplevel(0,axis=1)
    data = data[data['TOR60']>0]
    data['label_class'] =xs_qcut(data['LABEL0'],10)#.groupby(level='datetime').apply(lambda x:xs_qcut(x,10)) 
   


    print('==')
        