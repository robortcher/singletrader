"""
构造基于qlib.bin数据格式的因子类
"""

import logging
from numpy import iterable
from qlib.data import D
import qlib
from collections import OrderedDict
import datetime
import pandas as pd
from singletrader import __date_col__,__symbol_col__


__all__ = ['BaseFactor','FormatFactor','SigleFactor','MultiFactor','F', 'ModelFactor']
__bar_fields__ = ['$open', '$close', '$high', '$low', '$money', '$volume','$market_cap','$avg','$paused']
__bar_names__ = ['open', 'close', 'high', 'low', 'money', 'volume','market_cap','avg','paused']


class BaseFactor(): 
    """
    默认数据起始日期:2005-01-01
    默认数据结束日期:当天
    默认股票池:全市场
    默认频率:日频
    """
    default_start_date = '2005-01-01'
    default_end_date = datetime.datetime.now().strftime("%Y-%m-%d")
    default_instruments = 'all'
    default_freq = 'day'


    # @classmethod
    # def get_initial_field(cls):
    #     return get_initial_fields()

    def __init__(self,**kwargs):
        self._field=[]
        self._name = []
        self.bardata = None
        self.__data = None
        self.instruments = kwargs.get('instruments', self.default_instruments)
        self.start_date = kwargs.get('start_date', self.default_start_date)
        self.end_date = kwargs.get('end_date',self.default_end_date)
        self.freq = kwargs.get('freq',self.default_freq)
        # self.setup_data(**kwargs)

    def setup_data(self, **kwargs):#instruments, start_date,end_date,freq):
        instruments = kwargs.get('instruments', self.instruments)
        if instruments is None:
            instruments = self.default_instruments
        start_date = kwargs.get('start_date', self.start_date)
        end_date = kwargs.get('end_date',self.end_date)
        freq = kwargs.get('freq',self.freq)
        if type(instruments) is str:
            try:
                qlib_instruments = D.instruments(instruments)
                instruments =  D.list_instruments(instruments=qlib_instruments, start_time=start_date, end_time=end_date, as_list=True)
            except ValueError:
                instruments = [instruments]

        df = D.features(instruments, self._field, start_time=start_date, end_time=end_date, freq=freq).swaplevel(0,1)
        df.columns = self._name
        # bar_df =  D.features(instruments, __bar_fields__, start_time=start_date, end_time=end_date, freq=freq)
        # bar_df.columns = __bar_names__
        
        self.__data = df if self._name.__len__() > 1 else df[df.columns[0]]
        # self.bardata = bar_df
        self.__data.index =  self.__data.index.set_names([__date_col__,__symbol_col__])
        return self.__data

    @property
    def _data(self):
        if self.__data is None:
            logging.info("开始生成因子数据...")
            return self.setup_data()
        
        elif type(self.__data) in (pd.DataFrame,pd.Series):
            return self.__data
    
    def get_stock_next_return(self,price="$open",add_shift=1,periods=(1,),**kwargs):
        instruments = kwargs.get('instruments', self.default_instruments)
        start_date = kwargs.get('start_date', self.default_start_date)
        end_date = kwargs.get('end_date',self.default_end_date)
        freq = kwargs.get('freq',self.default_freq)
        if type(instruments) is str:
            try:
                qlib_instruments = D.instruments(instruments)
                instruments =  D.list_instruments(instruments=qlib_instruments, start_time=start_date, end_time=end_date, as_list=True)
            except ValueError:
                instruments = [instruments]
        if type(periods) is int:
            periods = (periods,)
        shift = add_shift
        return_fields = [f"Ref(Ref({price}, -{d}) / {price} - 1, -{shift})" for d in periods]
        return_names = [f'return{d}D' for d in periods]
        return_data = D.features(instruments, return_fields, start_time=start_date, end_time=end_date, freq=freq)
        return_data.columns = return_names
        if kwargs.get('method',None) == 'zscore':
            return_data = return_data.groupby(level='datetime').apply(lambda x:(x-x.mean()) / x.std())
        elif kwargs.get('method',None) == 'rank':
            return_data = return_data.groupby(level='datetime').rank()
        return return_data


class FormatFactor(BaseFactor):
    def __init__(self, expression, params, name=None,**kwargs):
        super().__init__(**kwargs)
        if iterable(params):
            for param in params:
                _expression = expression.format(param)
                self._field.append(_expression)
                self._name.append(_expression if name is None else (name+str(param)))
        else:
            _expression = expression.format(params)
            self._field.append(_expression)
            self._name.append(expression if name is None else (name+param))
        

class SigleFactor(BaseFactor):
    def __init__(self,field:str or list or dict, 
            name:str or list=None,
            **kwargs
        ) -> None:
        super().__init__(**kwargs)
        if type(field) is str:
            field = [field]
    
        elif type(field) is dict:
            _field = OrderedDict(field)
            name = list(_field.keys())
            field = list(_field.values())
         
        if name is None:
            name = field
        
        else:
            if type(name) is str:
                name = [name]    
        
        self._field += field
        self._name += name
        

class MultiFactor(BaseFactor):
    default_field = []
    default_name = []
    def __init__(self,field,name=None,**kwargs) -> None:
        super().__init__(**kwargs)
        self.add_features(field, name)
    
    def add_features(self, field:str or list or dict, name:str or list=None, refresh=False):
        if refresh==False:
            self._field = self.default_field.copy()
            self._name = self.default_name.copy()
        
        if type(field) is str:
            field = [field]
        
        elif type(field) is dict:
            _field = OrderedDict(field)
            name = list(_field.keys())
            field = list(_field.values())
         
        if name is None:
            name = field
        
        else:
            if type(name) is str:
                name = [name]    
        
        self._field += field
        self._name += name


class ModelFactor:
    def __init__(self,
        config={},
        **kwargs
    ):
        """
        features:dict{'name':[], field:[]}
        labels:dict{'name':[], field:[]}
        train_interval:(start_date, end_date)
        """
        train_interval = kwargs.get('train', config.get('train',None))
        valid_interval = kwargs.get('valid', config.get('valid',None))
        test_interval = kwargs.get('test', config.get('test',None))
        _start_date = train_interval[0]
        _end_date = test_interval[1]
        kwargs['start_date'] = _start_date
        kwargs['end_date'] = _end_date
        features = kwargs.get('features', config.get('features',None))
        labels = kwargs.get('label', config.get('label',None))
        feature_factor = MultiFactor(name=features['name'], field=features['field'],**kwargs)
        label_factor = MultiFactor(name=labels['name'], field=labels['field'],**kwargs)
        processors = kwargs.get('processor',config.get('processor',{'features':[],'label':[]}))
        self.raw_data = pd.concat({'features':feature_factor._data, 'label':label_factor._data},axis=1)
        self._data = self.raw_data.copy()
        self.data_l={}
        f_process = processors.get('features',[])
        l_process = processors.get('label',[])
        for process in f_process:
            self._data['features'] = process(self._data['features'])
        for process in l_process:
            self._data['label'] = process(self._data['label'])

        # dataset_split = kwargs.get('split', config.get('split', None))
        # if dataset_split is not None:
        

        if train_interval is not None:
            self.data_l['train'] = self._data[(self._data.index.get_level_values('datetime')>=train_interval[0]) & 
                (self._data.index.get_level_values('datetime')<=train_interval[1])
            ]
        if valid_interval is not None:
            self.data_l['valid'] = self._data[(self._data.index.get_level_values('datetime')>=valid_interval[0]) & 
                (self._data.index.get_level_values('datetime')<=valid_interval[1])
                ]     
        if test_interval is not None:
            self.data_l['test'] = self._data[(self._data.index.get_level_values('datetime')>=test_interval[0]) & 
                (self._data.index.get_level_values('datetime')<=test_interval[1])
                ]
    
    def prepare(self,dataset_name=None,return_tensor=False):
        if dataset_name is None or self.data_l=={}:
            return self._data
        else:
            try:
                return self.data_l[dataset_name]
            except KeyError:
                return None
F=SigleFactor
    

if __name__ == '__main__':
    
    root_dir = r'/data0/xmh/.xlib'
    QLIB_BIN_DATA_PATH = root_dir+'/qlib_data'
    # from ..constant import QLIB_BIN_DATA_PATH
    qlib.init(provider_uri=QLIB_BIN_DATA_PATH)
    test_field = {"ROC%d"%d:"Ref($close,%d)" % d for d in (1,5,10,20,30,60)}
    fc = SigleFactor(test_field)
    fc.get_stock_next_return()
    # fc.setup_data()
    print