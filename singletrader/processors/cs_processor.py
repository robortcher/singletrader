from functools import partial
from .normal_processor import winzorize
from abc import abstractmethod
from singletrader import __symbol_col__ ,__date_col__


class CsProcessor():
    def __init__(self,**kwargs):
        self.kwargs = kwargs

    def __call__(self, df):
        data =  df.groupby(level=__date_col__).apply(self.func)
        return data
    



    @property
    def func(self):
        pass


class CsWinzorize(CsProcessor):
    def __init__(self,k=5,method='sigma'):
        self.k = k
        self.method = method
    
    @property
    def func(self):
        return partial(winzorize,k=self.k,method=self.method)
  
