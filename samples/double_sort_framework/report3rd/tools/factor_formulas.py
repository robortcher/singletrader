"""
计算因子相关属性的函数包
包括:
    构造纯因子组合权重计算；
    分组收益；
    因子ic值；
    因子中性化处理；

"""
import numpy as np
import pandas as pd
from functools import partial
import logging
try:
    from constant import Ind_info
except:
    from .constant import Ind_info
# from dataprocessing.MarketConst import Ind_info
# from factor_building.constant import get_industry_info,Ind_info

def xs_correlation(x, y):
	if type(x.index) is not pd.MultiIndex:
		x = x.stack()
	if type(y.index) is not pd.MultiIndex:
		y = y.stack()
	xy = pd.DataFrame(x)
	xy['y'] = y
	xy_corr = xy.groupby(level=0).apply(lambda x: x.corr().iloc[0, 1])
	return xy_corr

def get_pure_factor_portfolio(factor_data, zero_capital=False, 
                only_long=False, threshold=0.0, periods=1):
    """
    计算纯因子组合权重
    parameters
    facto_data : pd.Series, Mutil-index:(tradeTime:datetime, symbol:str)
    """
    def cw(data, only_long=False, threshold=0.0):
        data = data.T
        try:
            data_without_nan = data.dropna()
            data_without_nan.insert(loc=0, value=1,column='const')
            # data_without_nan = sm.add_constant(data_without_nan)
            inv = np.linalg.inv(data_without_nan.T.dot(data_without_nan))
            values = inv[1].dot(data_without_nan.T)
            res = pd.DataFrame(values, index = data_without_nan.index,columns = data.columns)
            res = pd.DataFrame(np.where(res.abs() > threshold, res, 0), index = res.index, columns= res.columns)

            if only_long:
                res = pd.DataFrame(np.where(res > 0, res, 0), index = res.index, columns= res.columns)
            res = res.reindex(data.index)
            return res
        
        except Exception as e:
            logging.error(e)     
            res = pd.DataFrame(np.nan, index=data.index, columns=data.columns)
            return res
    
    data = factor_data.unstack()
    cw = partial(cw, only_long=only_long, threshold=threshold)
    raw_weight = data.groupby(level=0).apply(cw)
    raw_weight = raw_weight.unstack().droplevel(0, axis=1)
    if zero_capital:
        pass
        # raw_weight = raw_weight.apply(unit_transfrom,axis=1)
        # raw_weight = (raw_weight.T / (data * raw_weight).sum(axis=1)).T
    
    period_weight = raw_weight[::periods]
    period_weight = period_weight.reindex(raw_weight.index).ffill()
    return period_weight

def get_group_returns(factor_data, ret_data, groups=5, holding_period=1, cost=0,is_group_factor=False,return_weight=False):
    """
    分组收益计算
    """
    # factor_group = factor_data.dropna().groupby(level= 0).apply(lambda x:pd.qcut(x, groups, duplicates='drop', labels=list(range(groups))))
    def _qcut(data, n=groups):
        try:
            return pd.qcut(data.rank(method='first'),n,duplicates='drop',labels=list(range(n)))
        except:
            return pd.Series(np.nan,index = data.index,name=data.name)
    
    if is_group_factor:
        factor_group = factor_data.dropna()
    else:
        factor_group = factor_data.dropna().groupby(level= 0).apply(_qcut)
    
    if holding_period > 1:
        factor_group_unstack = factor_group.unstack()
        factor_group_unstack = factor_group_unstack[::holding_period].reindex(factor_group_unstack.index).ffill()
        factor_group = factor_group_unstack.stack()

    factor_group = factor_group.reindex(ret_data.index)
    group_change = factor_group.sort_index().unstack().astype(np.float).fillna(-1).diff().fillna(0)
    #####开始计算不同组别权重########
    
    group_change[group_change.abs()>0] = 1
    group_change = group_change.stack()
    group_change = group_change.reindex(factor_group.index)
    ret_data_raw = ret_data.copy()
    ret_data = ret_data - cost*group_change #
    

    if not factor_group.dropna().empty:
        group_rets = ret_data.groupby(factor_group).apply(lambda x:x.groupby(level=0).mean()).swaplevel(0,1).sort_index()
        group_rets = group_rets.unstack()
        group_rets_raw = ret_data_raw.groupby(factor_group).apply(lambda x:x.groupby(level=0).mean()).swaplevel(0,1).sort_index()
        group_rets_raw = group_rets_raw.unstack()  
        
        group_rets['hml'] = group_rets_raw.iloc[:,-1] - group_rets_raw.iloc[:,0]
        group_rets['lmh'] = group_rets_raw.iloc[:,0] - group_rets_raw.iloc[:,-1]
        group_rets = group_rets.stack() 
    else:
        return None
        # group_rets = factor_group
    group_rets.name = factor_data.name
    if return_weight:
        group_weights = factor_group.astype(np.float).groupby(factor_group).apply(lambda x:x.unstack().apply(lambda x:(x+1)/(x+1).sum(),axis=1)).fillna(0)#.stack()
        hml_weight = group_weights[group_weights.index.get_level_values(0)==(groups-1)].droplevel(0) - group_weights[group_weights.index.get_level_values(0)==0].droplevel(0) 
        hml_weight[factor_data.name] = 'hml'
        hml_weight = hml_weight.set_index(factor_data.name,append=True).swaplevel(0,1)

        lmh_weight = group_weights[group_weights.index.get_level_values(0)==0].droplevel(0) - group_weights[group_weights.index.get_level_values(0)==(groups-1)].droplevel(0) 
        lmh_weight[factor_data.name] = 'lmh'
        lmh_weight = lmh_weight.set_index(factor_data.name,append=True).swaplevel(0,1)
        
        
        
        group_weights = pd.concat([group_weights,hml_weight,lmh_weight])
        #group_weights.name = factor_data.name
        return group_rets, group_weights
    return group_rets



def get_factor_ic(factor_data, ret_data, method='normal',universe=None):
    """
    获取因子收益
    """
    # from factor_building.qformulas import xs_correlation
    if universe is not None:
        # factor_data = factor_data[universe]
        factor_data = factor_data.reindex(universe.index)[universe]
        ret_data = ret_data.reindex(universe.index)[universe]
        # ret_data = ret_data[universe]
    if method == 'rank':
        factor_data = factor_data.groupby(level=0).rank()
        ret_data = ret_data.groupby(level=0).rank()
    
    ic_df = xs_correlation(factor_data, ret_data)
    ic_df.name = factor_data.name
    return ic_df


def industry_neutralize(factor_data, ind_info=None, name='sw_l1'):
    """
    行业中性化处理
    """
    
    if factor_data.index[0].__len__()==2:
        factor_data = factor_data.droplevel(0)
    if ind_info is None:
        ind_info = Ind_info
        #iind_info = get_industry_info()
    ind_info = ind_info.reindex(factor_data.index)
    factor_data = factor_data.groupby(ind_info).apply(lambda x:x-x.mean())
    return factor_data

def get_predict_resid(data, add_constant=True, y_loc=0):
    """
    获取数据集的预测值和残差
    默认第一列为被解释变量，其余为解释变量
    """
    import statsmodels.api as sm
    if data.index[0].__len__()==2:
        data = data.droplevel(0)
    
    ret_data  = data.iloc[:, y_loc]
    factor_data = pd.concat([data.iloc[:, :y_loc],data.iloc[:, y_loc+1:]],axis=1)

    if add_constant:
        factor_data = sm.add_constant(factor_data)

    xy = pd.concat([factor_data,ret_data],axis=1).dropna()
    if xy.__len__()==0:
        return None
    model = sm.OLS(xy.iloc[:,-1], xy.iloc[:,:-1]).fit()
    predict_data = pd.DataFrame(index=xy.index, data = np.c_[model.predict(), model.resid])
    predict_data.columns = ['pred','resid']
    return predict_data

def get_resid(data, add_constant=True, y_loc=0):
    """
    获取数据集的预测值和残差
    默认第一列为被解释变量，其余为解释变量
    """
    import statsmodels.api as sm
    if data.index[0].__len__()==2:
        data = data.droplevel(0)
    
    ret_data  = data.iloc[:, y_loc]
    factor_data = pd.concat([data.iloc[:, :y_loc],data.iloc[:, y_loc+1:]],axis=1)

    if add_constant:
        factor_data = sm.add_constant(factor_data)

    xy = pd.concat([factor_data,ret_data],axis=1).dropna()
    if xy.__len__()==0:
        return pd.Series(index=xy.index, data = np.nan,name=data.columns[y_loc])
    model = sm.OLS(xy.iloc[:,-1], xy.iloc[:,:-1]).fit()
    predict_data = pd.Series(index=xy.index, data=model.resid,name=data.columns[y_loc])
    # predict_data.columns = ['pred','resid']
    return predict_data


def get_factor_resid_n(data,n=1,add_constant=True):
    """
    n为y变量个数
    """
    y_columns = data.columns.tolist()[:n]
    x_columns = data.columns.tolist()[n:]
    datas = [data[[factor_name]+x_columns] for factor_name in y_columns]
    res = list(map(get_resid,datas))
    return pd.concat(res,axis=1)




def get_betas(X,Y, add_constant=True,groupby_loc=0):
    """
    矩阵求解
    """
    import statsmodels.api as sm
    Y = pd.DataFrame(Y)
    if add_constant:
        X = sm.add_constant(X)
    X_names = X.columns
    Y_names = Y.columns
    XY = pd.concat([X,Y],axis=1).dropna()
    def __get_betas(data):
        if data.index[0].__len__() == 2:
            data = data.droplevel(groupby_loc)
        X = data[X_names]
        Y = data[Y_names]
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        beta = pd.DataFrame(beta.T, index=Y.columns,columns=X.columns)
        return beta
    if XY.index[0].__len__() == 2:
        # data = data.droplevel(groupby_loc)
        predict = XY.groupby(level=groupby_loc).apply(__get_betas)
    else:
        predict = __get_betas(XY)
    return predict

def get_factor_beta_n(data,n=1,add_constant=True):
    """
    n为y变量个数
    """
    y_columns = data.columns.tolist()[:n]
    x_columns = data.columns.tolist()[n:]
    datas = [data[[factor_name]+x_columns] for factor_name in y_columns]
    res = list(map(get_beta,datas))
    return pd.concat(res,axis=1)

def get_beta(data, add_constant=True, y_loc=0):
    """
    获取数据集的指定beta
    默认第一列为被解释变量，其余为解释变量
    ***后期考虑和get_predict_resid函数合并，提高效率
    """
    import statsmodels.api as sm
    if data.index[0].__len__()==2:
        data = data.droplevel(0)
    ret_data  = data.iloc[:, y_loc]
    factor_data = pd.concat([data.iloc[:, :y_loc],data.iloc[:, y_loc+1:]],axis=1)
    if add_constant:
        factor_data = sm.add_constant(factor_data)
    xy = pd.concat([factor_data,ret_data],axis=1).dropna()
    if xy.__len__()==0:
        return None
    model = sm.OLS(xy.iloc[:,-1], xy.iloc[:,:-1]).fit()
    beta_data = model.params
    return beta_data


def get_beta2(data, add_constant=True, y_loc=0):
    """
    获取数据集的指定beta
    默认第一列为被解释变量，其余为解释变量
    ***后期考虑和get_predict_resid函数合并，提高效率
    """
    import statsmodels.api as sm
    if data.index[0].__len__()==2:
        data = data.droplevel(0)
    ret_data  = data.iloc[:, y_loc]
    factor_data = pd.concat([data.iloc[:, :y_loc],data.iloc[:, y_loc+1:]],axis=1)
    if add_constant:
        factor_data = sm.add_constant(factor_data)
    xy = pd.concat([factor_data,ret_data],axis=1).dropna()
    if xy.__len__()==0:
        return None
    model = sm.OLS(xy.iloc[:,-1], xy.iloc[:,:-1]).fit()
    beta_data = model.params
    resid_data = model.resid
    rsquared_data = model.rsquared
    adj_rsquared_data =  model.rsquared_adj
    return beta_data,resid_data#,rsquared_data, adj_rsquared_data




def add_ind_dummies(factor_data, ind_info=None):
    """
    添加行业哑变量
    factor_data.index必须是聚宽的股票代码结构
    数据来源：jqdata
    """
    if factor_data.index[0].__len__()==2:
        factor_data = factor_data.droplevel(0)  ### 需要改进
    if ind_info is None:
        ind_info = Ind_info
    ind_info = ind_info.reindex(factor_data.index)
    ind_dummies = pd.get_dummies(ind_info)
    factor_data = pd.concat([factor_data,ind_dummies],axis=1)
    #factor_data = factor_data.groupby(ind_info).apply(lambda x:x-x.mean())
    return factor_data

def standardize(factor_data, method='std'):
    """
    截面标准化处理
    Parameters
    factor_data:pd.DataFrame
                Multi_Index(date:str or datetime, symbol:str)
    method:str,'std','rank', 'rank_ratio' 
    """
    if type(factor_data.index) is not pd.MultiIndex:
        factor_data = factor_data.stack()
    if factor_data.index[0].__len__() == 2:
        factor_data = factor_data.droplevel(0)  ### 需要改进
    if method == 'std':
        factor_data = (factor_data - factor_data.mean()) / factor_data.std()
    elif method == 'rank':
        factor_data = factor_data.rank()
    elif method == 'rank_ratio':
        factor_data = factor_data.rank() / factor_data.rank().max()
    return factor_data


def winzorize(factor_data,k=5,method='sigma'):
    """
    极值化处理
    k: float or shape(1,2) iterable
    method: str 'sigma','mad','qtile'
    """
    x = factor_data.droplevel(0)
    # x = pd.DataFrame(x)
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
    
    y = pd.DataFrame(y, index = x.index, columns = x.columns)
    return y

def xs_qcut(data, n=5, max_loss=0.2):
    """截面分组"""
    length = data.__len__()
    def _qcut(data, n=n):
        try:
            res = pd.qcut(data,n,duplicates='drop',labels=list(range(n)))
        except:
            res =  pd.Series(np.nan,index = data.index,name=data.name)
        return res
    
    if type(data) is pd.DataFrame:
        
        res = data.dropna().groupby(level= 0).apply(lambda x:x.apply(lambda x:_qcut(x)))
        for col in res.columns:
            if res.dropna().__len__() / length < (1 - max_loss):
                logging.info("{} loss is more than {}".format(col, max_loss))
                res = res.drop(col, axis=1) 
        return res
    else:
        return _qcut(data)


def bardata_resample(data, freq=5):
    """数据降采样，按指定频率和成k线"""
    data['open'] = data['open'].shift(freq-1)
    data['high'] = data['high'].rolling(freq).max()
    data['low'] = data['low'].rolling(freq).min()
    data['volume'] = data['volume'].rolling(freq).sum()
    data['money'] = data['money'].rolling(freq).sum()
    data = data.iloc[freq-1::freq]
    return data