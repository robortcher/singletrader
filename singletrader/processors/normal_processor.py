import numpy as np
from singletrader import __symbol_col__,__date_col__
import pandas as pd


def check_and_delete_level(df,delete_level=__date_col__):
    if isinstance(df.index,pd.MultiIndex):
        df = df.droplevel(delete_level)
    return df

def winzorize(factor_data,k=5,method='sigma'):
    """
    极值化处理
    k: float or shape(1,2) iterable
    method: str 'sigma','mad','qtile'
    """
    x = check_and_delete_level(factor_data)
    if method == 'mad':
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0)
        uplimit = med + k *mad
        lwlimit = med - k* mad
        y = np.where(x >= uplimit, uplimit, np.where(x <=lwlimit, lwlimit, x))

    elif method == 'sigma':
        me = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        uplimit = me + k * sigma
        lwlimit = me - k* sigma
        y = np.where(x >= uplimit, uplimit, np.where(x <=lwlimit, lwlimit, x))

    elif method == 'qtile':
        uplimit = np.quantile(x, q = k[1], axis=0)
        lwlimit = np.quantile(x, q = k[0], axis=0)
        y = np.where(x >= uplimit, uplimit, np.where(x <=lwlimit, lwlimit, x))
    
    elif method == 'qtile-median':
        if isinstance(k,float):
            k = (k,1-k)
        uplimit = np.quantile(x.dropna(), q = k[1], axis=0)
        lwlimit = np.quantile(x.dropna(), q = k[0], axis=0)
        y = np.where(x >= uplimit, x.median(), np.where(x <=lwlimit, x.median(), x))
    if isinstance(x,pd.Series):
        y = pd.Series(y,index=x.index,name=x.name)
    elif isinstance(x,pd.DataFrame):
        y = pd.DataFrame(y, index=x.index, columns=x.columns)
    return y