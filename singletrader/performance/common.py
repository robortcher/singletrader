# -*- coding: utf-8 -*-
"""
收益评价相关函数
"""
import pandas as pd
import numpy as np
def performance_indicator(nvs:pd.DataFrame, riskless_ret=0.02,freq=252,language='cn'):
    """给定指定净值或收盘价计算相关投资收益评估指标"""
    nvs = pd.DataFrame(nvs)
    def __performance_indicator(nv:pd.Series, riskless_ret):
        #装载累计收益
        nv_df = nv.ffill().dropna()
        nv_df = nv_df.bfill()
        portfolio_ret = nv_df / nv_df.iloc[0]

        #计算每日收益
        portfolio_ret_daily = portfolio_ret.pct_change().fillna(0)
        days_number = nv.__len__()
        
        #计算各类指标
        abs_ret = portfolio_ret.iloc[-1] - 1
        ret_ann = (abs_ret+1) ** (freq / (len(nv_df)-1)) - 1
        sigma_ann = portfolio_ret_daily.std()*(freq**0.5)

        # ret_yoy = (abs_ret+1) ** (250 / len(nv_df)) - 1
        max_drawdown = ((portfolio_ret.cummax() - portfolio_ret)/portfolio_ret.cummax()).max()
        
        ann_return1Y = nv_df.iloc[-1] / nv_df.iloc[-min((freq*1+1),nv_df.__len__())] -1
        ann_return2Y = (nv_df.iloc[-1] / nv_df.iloc[-min((freq*2+1),nv_df.__len__())]) **(1/(min(freq*2,nv_df.__len__()-1)/freq))-1
        ann_return3Y = (nv_df.iloc[-1] / nv_df.iloc[-min((freq*3+1),nv_df.__len__())]) **(1/(min(freq*3,nv_df.__len__()-1)/freq))-1,
        ann_return5Y = (nv_df.iloc[-1] / nv_df.iloc[-min((freq*5+1),nv_df.__len__())]) **(1/(min(freq*5,nv_df.__len__()-1)/freq))-1,
        
        if max_drawdown == 0:
            calmar_ratio = 0
        else:
            calmar_ratio = ret_ann/max_drawdown
        
        sharpe_ratio = (ret_ann - riskless_ret) / sigma_ann 

        if language.lower() == 'en':
            _performance_indicator = {'abs_ret'    : abs_ret,
                                    'ret_ann'    : ret_ann,
                                    'sigma_ann'  : sigma_ann,
                                    'max_drawdown':max_drawdown,
                                    f'SR rf={riskless_ret}':sharpe_ratio,
                                    'calmar_ratio':calmar_ratio,
                                    'days_number':days_number,
                                    # 'ann.return 1Y ':ann_return1Y,
                                    # 'ann.return 2Y':ann_return2Y,
                                    # 'ann.return 3Y':ann_return3Y,
                                    # 'ann.return 5Y':ann_return5Y,
                                    }
        elif language.lower() == 'cn':
            _performance_indicator = {'累计收益':abs_ret,
                            '年化收益':ret_ann,
                            '年化波动率':sigma_ann,
                            '最大回撤':max_drawdown,
                            f'夏普比({riskless_ret})':sharpe_ratio,
                            '年化收益/最大回撤':calmar_ratio,
                            '累计交易日':days_number,
                            # 'ann.return 1Y ':ann_return1Y,
                            # 'ann.return 2Y':ann_return2Y,
                            # 'ann.return 3Y':ann_return3Y,
                            # 'ann.return 5Y':ann_return5Y,
                            }
    
        _performance_indicator = pd.DataFrame(_performance_indicator,index= ['performance_indicator'])
        return _performance_indicator.T
    
    all_df = pd.concat([__performance_indicator(nvs[i],riskless_ret) for i in nvs.columns],axis=1)
    all_df.columns = nvs.columns
    return all_df

def cal_turnover_ratio(orders,net_asset_value,price_col='price_filled_avg',volume_col='volume_entrust'):
    """根据订单和每日净资产估算每日换手率
    """
    order_amount = orders[price_col] * orders[volume_col]
    amount_total = order_amount.groupby(level=0).sum()
    amount_buy = order_amount[orders['order_side'].isin([0])].groupby(level=0).sum()
    amount_sell = order_amount[orders['order_side'].isin([1])].groupby(level=0).sum()
    
    turnover_ratio_total = (amount_total / net_asset_value).fillna(0)
    turnover_ratio_buy = (amount_buy / net_asset_value).fillna(0)
    turnover_ratio_sell = (amount_sell / net_asset_value).fillna(0)
    
    turnover_ratio = pd.concat([turnover_ratio_total,turnover_ratio_buy,turnover_ratio_sell],axis=1).astype(np.float)
    turnover_ratio.columns = ['turnover_ratio_total','turnover_ratio_buy','turnover_ratio_sell']
    return turnover_ratio
    
    
