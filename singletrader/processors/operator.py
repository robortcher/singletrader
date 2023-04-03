import pandas as pd
import statsmodels.api as sm
import numpy as np
from singletrader import __date_col__,__symbol_col__
def get_beta(data, add_constant=True, y_loc=0,value='params'):
    """
    获取数据集的指定beta
    默认第一列为被解释变量，其余为解释变量
    ***后期考虑和get_predict_resid函数合并，提高效率
    """
    if isinstance(data.index,pd.MultiIndex):
    # if data.index[0].__len__()==2:
        data = data.droplevel(__date_col__)
    ret_data  = data.iloc[:, y_loc]
    factor_data = pd.concat([data.iloc[:, :y_loc],data.iloc[:, y_loc+1:]],axis=1)
    if add_constant:
        factor_data = sm.add_constant(factor_data)
    xy = pd.concat([factor_data,ret_data],axis=1).dropna()
    if xy.__len__()==0:
        return None
    model = sm.OLS(xy.iloc[:,-1], xy.iloc[:,:-1]).fit()
    res = getattr(model,value)
    return res
