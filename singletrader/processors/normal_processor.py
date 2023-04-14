import numpy as np
from singletrader import __symbol_col__,__date_col__
import pandas as pd

def industry_neutralize(factor_data, ind_info=None, name='sw_l1'):
    """
    行业中性化处理
    """
    if isinstance(factor_data.index,pd.MultiIndex):
    # if factor_data.index[0].__len__()==2:
        factor_data = factor_data.droplevel(0)
    if ind_info is None:
        from  singletrader.constant import Ind_info
        ind_info = Ind_info
        #iind_info = get_industry_info()
    ind_info = ind_info.reindex(factor_data.index)
    factor_data = factor_data.groupby(ind_info).apply(lambda x:x-x.mean())
    return factor_data

def check_and_delete_level(df,delete_level=__date_col__):
    if isinstance(df.index,pd.MultiIndex):
        df = df.droplevel(delete_level)
    return df

def _add_cs_data(tcs_data,cs_data,merge=True):
    
    """
    截面数据在时序上的填充
    merge: bool
    """
    if merge:
        data = tcs_data.groupby(level=__date_col__).apply(lambda x:pd.concat([x.droplevel(__date_col__),cs_data],axis=1)) 
    else:
        data = tcs_data.groupby(level=__date_col__).apply(lambda x:cs_data) 
    return data

def _add_ts_data(tcs_data,ts_data,merge=True):
    """截面数据在时序上的填充"""
    feature_index= tcs_data.index
    if merge:
        tcs_data = tcs_data.reset_index()
        ts_data = ts_data.reset_index()
        merge_data = pd.merge(tcs_data,ts_data,on=__date_col__,how='outer')
        merge_data = merge_data.set_index([__date_col__,__symbol_col__]).reindex(feature_index)
    else:
        merge_data = ts_data.reindex(tcs_data.index.get_level_values(__date_col__))
        merge_data.index = feature_index
    return merge_data#tcs_data.groupby(level=__symbol_col__).apply(lambda x:pd.concat([x.droplevel(__symbol_col__),ts_data],axis=1))

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
        if isinstance(k,float):
            k = (k,1-k)
        uplimit = np.quantile(x, q = max(k), axis=0)
        lwlimit = np.quantile(x, q = min(k), axis=0)
        y = np.where(x >= uplimit, uplimit, np.where(x <=lwlimit, lwlimit, x))
    
    elif method == 'qtile-median':
        if isinstance(k,float):
            k = (k,1-k)
        
        uplimit = np.quantile(x.dropna(), q = max(k), axis=0)
        lwlimit = np.quantile(x.dropna(), q = min(k), axis=0)
        y = np.where(x >= uplimit, x.median(), np.where(x <=lwlimit, x.median(), x))
    if isinstance(x,pd.Series):
        y = pd.Series(y,index=x.index,name=x.name)
    elif isinstance(x,pd.DataFrame):
        y = pd.DataFrame(y, index=x.index, columns=x.columns)
    return y

def standardize(data, method='z-score'):
    """
    截面标准化处理
    Parameters
    data:pd.DataFrame
                Multi_Index(date:str or datetime, symbol:str)
    method:str,'z-score','rank', 'rank_ratio' 
    """
    if method == 'z-score':
        data = (data - data.mean()) / data.std()
    elif method == 'rank':
        data = data.rank()
    elif method == 'rank_ratio':
        data = data.rank() / data.rank().max()
    return data