from functools import partial
from .normal_processor import winzorize,_add_cs_data,standardize
from abc import abstractmethod
import pandas as pd
from singletrader import __symbol_col__ ,__date_col__
from singletrader.processors.operator import get_beta

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


class CsNeutrualize(CsProcessor):
    def __init__(self, CN=True, SN=True):
        self.CN = True # 是否进行市值中性化
        self.SN = True # 是否进行行业中性化
        self.explain_data = None # 是否加载过数据
    def __call__(self, df,**kwargs):
        return self.func(data=df,**kwargs)
    
    def func(self,data,**kwargs):
        from singletrader.datasdk.qlib.base import MultiFactor
        from singletrader.constant import Ind_info
        
        # 时间戳判断和处理
        start_date = data.index.get_level_values(__date_col__).min()
        end_date = data.index.get_level_values(__date_col__).max()
        if hasattr(start_date,'strftime'):
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")
        
        if self.explain_data is not None:
            explain_data = self.explain_data
        else:
            explain_data = pd.DataFrame()
            if self.CN:
                explain_data = pd.concat([explain_data,MultiFactor(field=['Log($circulating_market_cap)'],name = ['market_cap'],start_date=start_date,end_date=end_date)._data],axis=1)
                explain_data.index = explain_data.index.set_names([__date_col__,__symbol_col__])
            if self.SN:    
                explain_data = _add_cs_data(explain_data,pd.get_dummies(Ind_info))
                explain_data.index = explain_data.index.set_names([__date_col__,__symbol_col__])
            self.explain_data = explain_data


        XY_data = pd.concat([data,explain_data],axis=1)
        resid_data = XY_data.groupby(level=__date_col__).apply(lambda x:get_beta(x,value='resid'))
        return resid_data


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
  
