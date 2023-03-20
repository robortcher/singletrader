from functools import partial
from .normal_processor import winzorize,_add_cs_data,standardize
from abc import abstractmethod
import pandas as pd
from singletrader import __symbol_col__ ,__date_col__


class CsProcessor():
    """截面处理器"""
    def __init__(self,**kwargs):
        self.kwargs = kwargs

    def __call__(self, df):
        data =  df.groupby(level=__date_col__).apply(self.func)
        return data
    



    @property
    def func(self):
        pass


class CsWinzorize(CsProcessor):
    """截面极值处理"""
    def __init__(self,k=5,method='sigma'):
        self.k = k
        self.method = method
    
    @property
    def func(self):
        return partial(winzorize,k=self.k,method=self.method)


class CsStandardize(CsProcessor):
    def __init__(self,method='z-score'):
        self.method = method


    @property
    def func(self):
        return partial(standardize,method=self.method)



class IndAggregation(CsProcessor):
    """截面行业聚合处理"""
    def __init__(self):
        """
        parameters
        ------------
        weights: pd.DataFrame,pd.Series,str
        ind_info: pd.DataFrame, pd.Series, str        
        """
    def __call__(self, df,**kwargs):
        return self.func(df,**kwargs)

    def func(self,data,weights='eq',ind_info=None):
        if isinstance(weights,str):
            if weights == 'eq':
                pass
            else:
                data['weights'] = data[weights]
        else:
            weights = pd.Series(weights)
            weights.name = 'weights'
            if isinstance(weights.index,pd.MultiIndex):
                data['weights'] = weights
            else:
                data = _add_cs_data(weights)
        
        if isinstance(ind_info,str):
            data =  data.rename(columns={ind_info:'ind_info'})
        
        else:
            ind_info = pd.Series(ind_info)
            ind_info.name = 'ind_info'
            if isinstance(ind_info.index,pd.MultiIndex):
                data['ind_info'] = ind_info
            else:
                data =  _add_cs_data(data,ind_info)
        if weights == 'eq':
            return data.groupby(level=__date_col__).apply(self._func_eq)
        else:
            return data.groupby(level=__date_col__).apply(self._func_cap)
    
    def _func_eq(self,data):
        return data.groupby('ind_info').mean()
    
    def _func_cap(self,data):
        df = data.groupby('ind_info').apply(self.__func2)
        return df
        # return data.groupby('ind_info').apply(lambda x:(x.apply(lambda x:x.drop('ind_info',axis=1)*x['weights'],axis=1) / x['weights'].sum()).sum())
    
    def __func2(self,data):
        return (data.drop('ind_info',axis=1).apply(lambda x:x*x['weights'],axis=1) / data['weights'].sum()).sum()
  
