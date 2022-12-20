"""

"""
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field

class BarDict(OrderedDict):
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            key = pd.to_datetime(key)
        return super().__getitem__(key)

@dataclass
class Bar():
    datetime:datetime or str
    _data:pd.DataFrame=field(default_factory=pd.DataFrame)
    symbol_col:str='symbol'
    def __post_init__(self):
        if self.symbol_col in self._data:
            self._data = self._data.set_index(self.symbol_col,drop=True)
        self._cols = self._data.columns
        for _c in self._cols:
            setattr(self,_c,self._data[_c])

class DataH:
    """
    get historical prices data from selected url or api
    """
    date_col = 'date'
    symbol_col = 'symbol'
    
    def __init__(self,src=None):
        """
        """

        self.src = src


    def download(self,symbol_list=None,start_date='2022-01-01',end_date='2022-01-30',just_data=False,**kwargs):       

        data = self._download_func(symbol_list=symbol_list,start_date=start_date,end_date=end_date,**kwargs)
        data = data.sort_values(by=[self.date_col,self.symbol_col])
        data[self.date_col] = pd.to_datetime(data[self.date_col])
        if not just_data:
            self.all_periods = sorted(set(data[self.date_col]))
            self.all_cols=data.columns[2:]
            self.all_symbols=sorted(list(set(data[self.symbol_col]))) 
            data = self._transform(data)
            self._data = data
        else:
            return self._transform(data)
            
    @property
    def _download_func(self):
        
        if type(self.src) is str:
            if self.src == 'jq':
                return download_jq
            elif self.src == 'yahoo':
                return download_yahoo
        elif isinstance(self.src,(pd.DataFrame,pd.Series)):
            return self._download_from_src


    
    def _download_from_src(self,**kwargs):
        return self.src
    
    def _transform(self,data):
        data = data.groupby('symbol').apply(lambda x:x.set_index(self.date_col).reindex(self.all_periods))
        del data[self.symbol_col]
        data = data.reset_index()
        # data[self.date_col] = data[self.date_col].astype(np.str)
        return data    
    
    @property
    def bar_generator(self):
        """generator of time bar"""
        group_by_date = self._data.set_index(self.date_col).groupby(level=0)
        
        group_size = group_by_date.size()
        all_periods = group_size.index.tolist()
        
        
        for _date,_bar in group_by_date:
            yield Bar(_date, _bar)


def download_jq(symbol_list,start_date,end_date,**kwargs):
    import jqdatasdk as jq
    # jq.auth(kwargs['user', kwargs['password']])
    jq.auth('15111126561','Wbzg207182')
    symbol_list = symbol_list if symbol_list is not None else jq.get_all_securities().index.tolist()
    data = jq.get_price(security=symbol_list,start_date=start_date,end_date=end_date, fields=['open', 'close', 'high','low','volume','paused'],panel=False,**kwargs)
    data = data.rename(columns = {'time':DataH.date_col, 'code':DataH.symbol_col})
    return data


def download_yahoo(symbol_list,start_date,end_date,**kwargs):
    import yfinance as yf
    data = yf.download(tickers=symbol_list, start=start_date, end=end_date,**kwargs)
    if len(symbol_list)>1:
        data = data.stack().drop('Close',axis=1)
    else:
        data['symbol'] = symbol_list[0]
        data = data.set_index('symbol',append=True)
    data.index = data.index.set_names([DataH.date_col,DataH.symbol_col])
    data.columns = [_c.lower() for _c in data.columns]
    data = data.rename(columns = {'adj close':'close'}).reset_index()
    return data


def bar_resample(data,frequency,symbol_level=1):
    
    data_output = {}
    data_output['close'] = data['close'].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).last())
    data_output['open'] = data['open'].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).first())
    data_output['high'] = data['high'].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).max())
    data_output['low'] = data['low'].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).min())
    
    # data_output['volume'] = data['volume'].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).sum())
    # data_output['volume'] = data['volume'].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).sum())
    
    
    data_output = pd.concat(data_output,axis=1)
    return data_output

if __name__ == '__main__':
    import logging
    dh_yf = DataH(src='yahoo')
    symbol_us = ['IVV','SCHD','AGG','SCHP']
    dh_yf.download(symbol_us)
    bar_generator = dh_yf.bar_generator
    while True:
        print(next(bar_generator))
        logging.info("======over======")
    
    dh_jq = DataH(src='jq')
    symbol_cn = ['000001.XSHE','600000.XSHG']
    dh_jq.download(symbol_cn)
    logging.info("======over======")
    

        