import logging
import pandas as pd
import numpy as np
from .functions import *
from singletrader.shared.utility import parLapply
from singletrader.performance.common import performance_indicator

from functools import partial
from itertools import product


import plotly.express as px
from plotly.figure_factory import create_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from singletrader import __date_col__,__symbol_col__

class FactorEvaluation:  
    """
    因子组合构建
    """
    def __init__(self, bar_data, factor_data,freq=252,winzorize=True,standardize=True,industry_neutralize=True):
        bar_data = pd.DataFrame(bar_data)
        factor_data = pd.DataFrame(factor_data)
        self.bar_data = bar_data
        self.raw_factor_data = factor_data.copy()
        self.infer_factor_data = self.raw_factor_data.copy()
        self.factor_data = self.infer_factor_data.copy()
        
        if winzorize:
            self.winzorize()
        if standardize:
            self.standardize()
        if industry_neutralize:
            self.industry_neutralize()
        self.freq=freq
        # self.all_factors = factor_data.columns
    
    @property
    def all_factors(self):
        return self.factor_data.columns.tolist()

    
    def get_factor_weight(self, factor_list=None, zero_capital=False, only_long=False, threshold=0.0, holding_period=1):
        """
        存储因子权重
        """
        if factor_list is None:
            factor_list  = self.all_factors
        weights_df = {}
        for factor in factor_list:
            factor_data = self.factor_data[factor]
            weights_df[factor] = get_pure_factor_portfolio(factor_data, zero_capital, only_long, threshold,holding_period)
        return weights_df

    # def get_next_return_(self, next_n=1, add_shift=1, base='open', neutralize=False,excess_return=False,bar_data=None,total=True):
    #     """
    #     获取未来指定某周期收益 
    #     """
    #     if bar_data is None:
    #         bar_data = self.bar_data
        
    #     price = self.bar_data[base].unstack()
    #     if total:
    #         next_return = (price.shift(-next_n) / price - 1).shift(-add_shift)
        
    #     else:
    #         next_return = (price.shift(-next_n) / price.shift(-next_n+1) - 1).shift(-add_shift)
        
    #     if neutralize:
    #         xy_data = pd.concat([next_return.stack(),pd.get_dummies(bar_data['industry_name'][bar_data['industry_name']!='nan']),bar_data['circulating_market_cap'].dropna()],axis=1)
    #         next_return = xy_data.groupby(level=0).apply(get_resid).unstack()
        
    #     if excess_return:
    #         next_return = next_return.apply(lambda x:x-x.mean(),axis=1)
    #     next_return.name = f'next{next_n}day_ret'
    #     setattr(self, f'next{next_n}day_ret', next_return)
    #     return next_return.stack()
    

    def get_next_return(self, next_n=1, add_shift=1,base='open', neutralize=False,excess_return=False,bar_data=None,total=True):
        """
        获取未来指定周期收益
        ----------
        Parameter

        next_n: int 未来某天
        """
        if bar_data is None:
            bar_data = self.bar_data

        #解析式表达拆分，目前只支持减号
        base = base.split('-') 
        


        if len(base) == 1:
            base = base[0]
            price = bar_data[base].unstack()
            
            if next_n == 0: # 0 期收益指定为0 
                return price / price - 1

            elif next_n>0:
                if total:
                    next_return = (price.shift(-next_n) / price - 1).shift(-add_shift)
                else:
                    next_return = (price.shift(-next_n) / price.shift(-next_n+1) - 1).shift(-add_shift)
            else:
                if total:
                    next_return = (price / price.shift(-next_n) - 1).shift(-add_shift)
                else:
                    next_return = (price.shift(-next_n-1) / price.shift(-next_n) - 1).shift(-add_shift)
        
        elif len(base)==2:
            price_t0 =  bar_data[base[1]].unstack()
            price_t1 =  bar_data[base[0]].unstack()
        
            if next_n == 0:
                return price_t0 / price_t0 - 1

            elif next_n >= 0:
                if total:
                    next_return = (price_t1.shift(-next_n) / price_t0.shift(-1) - 1).shift(-add_shift)
                
                else:
                    next_return = (price_t1.shift(-next_n) / price_t0.shift(-next_n) - 1).shift(-add_shift)
            else:
                if total:
                    next_return = (price_t1.shift(-1) / price_t0.shift(-next_n) - 1).shift(-add_shift)
                else:
                    next_return = (price_t1.shift(-next_n) / price_t0.shift(-next_n) - 1).shift(-add_shift)

        if neutralize:
            xy_data = pd.concat([next_return.stack(),pd.get_dummies(bar_data['industry_name'][bar_data['industry_name']!='nan']),bar_data['circulating_market_cap'].dropna()],axis=1)
            next_return = xy_data.groupby(level=0).apply(get_resid).unstack()
        
        if excess_return:
            next_return = next_return.apply(lambda x:x-x.mean(),axis=1)
        next_return.name = f'forward_return_{next_n}'
        setattr(self, f'forward_return_{next_n}', next_return)
        return next_return.stack()
        
    def get_factor_performance(self, 
            factor_list=None, 
            factor_weights=None, 
            next_n=1, 
            add_shift=1,
            zero_capital=False, 
            excess_return=False, 
            only_long=False,
            neutralize=False,
            threshold=0, 
            holding_period=1):
        """
        计算因子组收益、每日个股权重，每日仓位， 每日换手
        returns
        ---------
        all_returns_df:pd.DataFrame,各个因子的日收益
        factor_weights:dict {all_factors:factor_weights}
        all_turnover_df:pd.DataFrame,不同因子每日换手率
        position_ratios_df:pd.DataFrame,不同因子每日总仓位
        
        """
        if factor_list is None:
            factor_list  = self.all_factors
        
        if factor_weights is None:
            factor_weights = {}
        
        ret = self.get_next_return(next_n=next_n, neutralize=neutralize,add_shift=add_shift).unstack()
        if excess_return:
            ret = ret.apply(lambda x:x - x.mean(),axis=1)
        
        all_returns = []
        all_turnover_rates = []

        for factor in factor_list:
            logging.info(f"正在进行{factor}因子评估...")
            if factor not in factor_weights:
                factor_data = self.factor_data[factor]
                weights = get_pure_factor_portfolio(factor_data, zero_capital, only_long, threshold, holding_period)
                factor_weights[factor] = weights
            else:
                weights = factor_weights[factor]
        
            daily_return = (weights * ret).sum(axis=1)
            daily_return.name = factor
            all_returns.append(daily_return)
            
            turnover_rate = weights.diff()
            turnover_rate.iloc[0, :] = weights.iloc[0, :]
            turnover_rate = turnover_rate.abs().sum(axis=1)
            turnover_rate.name = factor
            all_turnover_rates.append(turnover_rate)
            logging.info(f"{factor}评估完成")
        all_returns_df = pd.concat(all_returns,axis=1)
        all_turnover_df = pd.concat(all_turnover_rates, axis=1)
        position_ratios_df = pd.concat([factor_weights[i].sum(axis=1) for i in factor_weights], axis=1)
        position_ratios_df.columns = factor_weights.keys()

        return all_returns_df,factor_weights,all_turnover_df,position_ratios_df

    def get_group_returns(self, factor_list=None, next_n=1, add_shift=1,groups=5, excess_return=False,base='open', start_date=None,end_date=None,cost=0,ret_data=None,holding_period=1, is_group_factor = False,return_weight=False,universe=None):
        if factor_list is None:
            factor_list = self.all_factors
        if ret_data is None:
            ret_data = self.get_next_return(next_n=next_n, base=base, add_shift=add_shift,excess_return=excess_return)
        group_results = []
        if universe is not None:
            universe = universe.reindex(self.factor_data.index)
        
        for factor in factor_list:
            # logging.info(f"正在计算因子{factor}的分组收益...")
            factor_data = self.factor_data[factor]
            if universe is not None:
                factor_data = pd.Series(np.where(universe>0,factor_data,np.nan),index=factor_data.index, name=factor)
 
            group_results.append(get_group_returns(factor_data, ret_data, groups=groups, holding_period=holding_period,cost=cost,is_group_factor=is_group_factor,return_weight=return_weight))
            # logging.info(f"因子{factor}的分组收益计算结束")
        if not return_weight:
            group_returns = pd.concat(group_results,axis=1)
            if start_date is not None:
                group_returns = group_returns[group_returns.index.get_level_values(0)>=start_date]
            if end_date is not None:
                group_returns = group_returns[group_returns.index.get_level_values(0)<=end_date]  
            return group_returns
        else:
            group_returns = pd.concat([i[0] for i in group_results],axis=1)
            if start_date is not None:
                group_returns = group_returns[group_returns.index.get_level_values(0)>=start_date]
            if end_date is not None:
                group_returns = group_returns[group_returns.index.get_level_values(0)<=end_date]  

            group_weights = {factor_list[i]:group_results[i][1] for i in range(factor_list.__len__())}
            turnover_ratio = {i:group_weights[i].groupby(level=0).apply(lambda x:x.diff().abs().sum(axis=1)).droplevel(0).unstack().T for i in group_weights}
            #group_weights = pd.concat([i[1] for i in group_results],axis=1)
            if start_date is not None:
                group_returns = group_returns[group_returns.index.get_level_values(0)>=start_date]
                group_weights = {i:group_weights[i][group_weights[i].index.get_level_values(level=__date_col__)>=start_date] for i in group_weights}
                turnover_ratio = {i:turnover_ratio[i][turnover_ratio[i].index.get_level_values(level=__date_col__)>=start_date] for i in turnover_ratio}
            if end_date is not None:
                group_returns = group_returns[group_returns.index.get_level_values(0)<=end_date]
                group_weights = {i:group_weights[i][group_weights[i].index.get_level_values(level=__date_col__)<=end_date] for i in group_weights}
                turnover_ratio = {i:turnover_ratio[i][turnover_ratio[i].index.get_level_values(level=__date_col__)<=end_date] for i in turnover_ratio}
            return group_returns,group_weights,turnover_ratio

    
    def get_factor_returns(self):
        pass



    def get_factor_ic(self,next_n=1,factor_list=None,base='open',method='normal',start_date=None,end_date=None,add_shift=1,excess_return=False,universe=None,total=True):
        freq_dict = {1:'Y',12:'M'}
        if factor_list is None:
            factor_list = self.all_factors
        ret_data = self.get_next_return(next_n=next_n, excess_return=excess_return,base=base,add_shift=add_shift,total=total)
        if universe is not None:
            universe = universe.reindex(self.factor_data.index)
        factor_ics = []
        # logging.info(f"正在计算因子{factor}的ic...")
        for factor in factor_list:
            factor_data = self.factor_data[factor]
            if universe is not None:
                factor_data = pd.Series(np.where(universe>0,factor_data,np.nan),index=factor_data.index, name=factor)
            factor_ics.append(get_factor_ic(factor_data, ret_data, method=method))
        # logging.info(f"因子{factor}的ic计算完毕")
        factor_ics = pd.concat(factor_ics,axis=1)
        # factor_ics = factor_ics.add_suffix(f"_{next_n}")
        if start_date is not None:
            factor_ics = factor_ics[start_date:]
        if end_date is not None:
            factor_ics = factor_ics[:end_date]
        self.factor_ics = factor_ics
        return factor_ics
    
    ###批量获取不同周期ic
    def __get_factor_ics(self,factor_list=None, periods=(1,), base='open',method='rank',start_date=None,end_date=None,add_shift=1,excess_return=False,universe=None, total=True):
        if factor_list is None:
            factor_list = self.all_factors
        all_columns = list(product(factor_list, periods))
        # all_columns = [i[0]+'_'+str(i[1])+'D' for i in all_columns]
        
        res = parLapply(periods, self.get_factor_ic,factor_list=factor_list,base=base,start_date=start_date,end_date=end_date,method=method,add_shift=add_shift,excess_return=excess_return,universe=universe,total=total)
        res = pd.concat(res,axis=1).dropna(how='all').sort_index(axis=1)
        # res = res.reindex(all_columns,axis=1)
        return res

    ###批量获取不同周期ic
    def get_factor_ics(self,periods=(1,),**kwargs):
        # if factor_list is None:
        #     factor_list = self.all_factors
        # all_columns = list(product(factor_list, periods))
        # all_columns = [i[0]+'_'+str(i[1])+'D' for i in all_columns]
        res = []
        for i in periods:
            df = self.get_factor_ic(next_n=i,**kwargs)
            df = df.add_suffix(f'_L{i}')
            res.append(df)
        res = pd.concat(res,axis=1).dropna(how='all')
        # res = res.reindex(all_columns,axis=1)
        return res


    def industry_neutralize(self):
        # logging.info("正在对因子作行业中性化处理...")
        self.factor_data = self.factor_data.groupby(level=0).apply(industry_neutralize)
        self.factor_data = self.factor_data.add_suffix('_GN')
        # logging.info("因子的行业中性化处理结束")
        return self.factor_data

    def standardize(self, method='std'):
        """
        Parameters
        method:str, 'std','rank','rank_ratio'
        """
        # logging.info("正在对因子作截面标准化处理...")
        __standardize = partial(standardize,method=method)
        self.factor_data = self.factor_data.groupby(level=0).apply(__standardize)
        # logging.info("因子的截面标准化处理结束")
        return self.factor_data

    def winzorize(self, k=3,method='sigma'):
        """
        极值化处理
        k: str or (2,1) iterable
        method: str 'sigma','mad','qtile'
        """
        # logging.info("正在对因子作截面去极值处理...")
        __winsorize = partial(winzorize,k=k,method=method)
        self.factor_data = self.factor_data.groupby(level=0).apply(__winsorize)
        # logging.info("因子的截面去极值处理结束")
        return self.factor_data

    def neutralize(self, industry_factor_name='industry_name'):
        """行业+市值中性化"""
        # logging.info("正在对因子作截面中性化处理...")
        bar_data = self.bar_data
        factor_data = self.factor_data
        factors_number = factor_data.__len__()
        get_factor_resid = partial(get_factor_resid_n,n=factors_number)
        factor_data = pd.concat([factor_data,pd.get_dummies(bar_data[industry_factor_name][bar_data['industry_name']!='nan']),bar_data['circulating_market_cap'].dropna()],axis=1)
        self.factor_data = factor_data.groupby(level=0).apply(get_factor_resid)
        # logging.info("截面中性化处理结束...")
        return factor_data

    def add_ind_dummies(self):
        # logging.info("正在对因子截面添加行业哑变量...")
        self.factor_data = self.factor_data.groupby(level=0).apply(add_ind_dummies)
        # logging.info("添加行业哑变量完毕")
        return self.factor_data
    
    def get_effective_factors(self, threshold=0.02, max_corr=0.7, start_date=None, end_date=None, **kwagrs):
        """
        获取有效因子
        Parameters
        threshold:float
        """
        if hasattr(self, 'factor_ics'):
            factor_ics = self.factor_ics
        else:
            factor_ics = self.get_factor_ic(**kwagrs)
        factor_ics = factor_ics[start_date:end_date]
        abs_ic = factor_ics.mean().abs().sort_values(ascending=False)
        abs_ic = abs_ic[abs_ic>threshold]

        if max_corr:
            corr = self.factor_data[abs_ic.index].corr().abs()
            useful_factor = []
            for factor in abs_ic.index:
                cur_corr = corr[factor][useful_factor]
                if useful_factor.__len__() == 0:
                    useful_factor.append(factor)
        
                else:
                    if cur_corr.max()  < max_corr:
                        useful_factor.append(factor)
            useful_ic = factor_ics.mean().reindex(useful_factor)
        else:
            useful_ic = factor_ics.mean().reindex(abs_ic.index)
        return useful_ic



    def get_hml_spred(self, factor_list=None, next_n=1, add_shift=1,groups=5, base='open', ret_data=None,holding_period=1, is_group_factor = False):
        gp = self.get_group_returns(factor_list=factor_list, next_n=next_n, add_shift=add_shift,groups=groups, base=base, ret_data=ret_data,holding_period=holding_period, is_group_factor=is_group_factor)
        gp = gp.groupby(level=0).apply(lambda x:x.iloc[-1]-x.iloc[0])
        return gp
    
    def get_summary(self,
                    start_date=None,
                    end_date=None,
                    groups=5,
                    excess_return=True,
                    holding_period=1,
                    base='close',
                    add_shift=1,
                    total=True,
                    universe=None,
                    ):
        """
        self: 自建类FactorEvaluation
        """
        def _ic_analysis(self):
            ic_df = self.get_factor_ic(method='normal',start_date=start_date,next_n=holding_period,base=base,add_shift=add_shift,excess_return=excess_return,total=total,universe=universe)
            result_df = pd.DataFrame(columns = ic_df.columns)
            result_df.loc['ic.mean'] = ic_df.mean()
            # result_df.loc['ic.ir'] = ic_df.mean() /  ic_df.std()
            period = holding_period
            result_df.loc['ic.t-stats'] = ic_df.mean() /  ic_df.std() * ((ic_df.__len__()) ** 0.5) /(period**0.5)
            # result_df.loc['rankic.mean'] = rankic_df.mean()
            # result_df.loc['rankic.ir'] = ic_df.mean() /  ic_df.std()
            # result_df.loc['rankic.t-stats'] = ic_df.mean() /  ic_df.std() * ((ic_df.__len__()/12) ** 0.5)
            return result_df


        # ic_summary
        ic_result = {}
        ic_result = _ic_analysis(self)


        # group returns
        # group_returns,group_weights,group_turnover = self.get_group_returns(holding_period=holding_period,start_date=start_date,end_date=end_date,excess_return=excess_return,base=base,add_shift=add_shift,groups=groups,return_weight=True)
        perfs = self.performance_summary(holding_period=holding_period,start_date=start_date,end_date=end_date,excess_return=excess_return,base=base,add_shift=add_shift,groups=groups,universe=universe)
        perfs = pd.concat(perfs)
        return_ann = perfs[perfs.index.get_level_values(1) == 'ret_ann'][['Long','Short','Long-Short']].unstack().T.swaplevel(0,1)
        TO = perfs[perfs.index.get_level_values(1) == 'turnover_ratio'][['Long','Short']].unstack().T.swaplevel(0,1)
        SR = perfs[perfs.index.get_level_values(1) == 'SR'][['Long','Short','Long-Short']].unstack().T.swaplevel(0,1)

        summary = round(pd.concat([ic_result,return_ann,SR,TO]),4)
        summary.index = ['ic.mean','ic.t-stats','AnnRet_Long','AnnRet_short','AnnRet','SR_Long','SR_Short','SR','TO_Long','TO_Short']
        #gp_summary
        # gp_result = self.get_group_returns(**kwargs)
        print(f'start_date:{start_date} / end_date:{end_date}')
        return summary
        

    def get_factor_detail_report(self,
                                factor,
                                ic_lags=(-4,13), 
                                base='close', 
                                groups=10,
                                holding_period=1,
                                add_shift=1,
                                total=True,
                                excess_return=False,
                                start_date=None,
                                end_date=None,
                                universe=None
                                ):
        
        factor_list=[factor]
        periods = list(range(ic_lags[0],ic_lags[1]))


        # get ic decay
        ic_series = self.get_factor_ic(add_shift=add_shift,base=base,factor_list=factor_list,start_date=start_date,end_date=end_date,universe=universe)
        
        # ic series
        ics_decay = self.get_factor_ics(periods=periods, factor_list=factor_list, base=base,add_shift=add_shift,total=total,start_date=start_date,end_date=end_date,universe=universe)

        
        # get group exccess return performance
        group_returns,group_weights,group_turnover = self.get_group_returns(holding_period=holding_period,universe=universe,start_date=start_date,end_date=end_date,excess_return=excess_return,factor_list=factor_list,base=base,add_shift=add_shift,groups=groups,return_weight=True)
        group_returns = group_returns[factor].unstack()
        

        #过滤掉Long-Short
        group_returns = group_returns.drop(['Long','Short'],axis=1)

        group_nvs = (group_returns+1).cumprod()
        group_nvs = group_nvs/group_nvs.iloc[0]
        group_nvs.name = 'holding_period=' + str(holding_period)

        group_turnover = group_turnover[factor].mean() / 2
        perfs = performance_indicator(group_nvs,ret_data=True,language='en',freq=self.freq)
        perfs.loc['turnover_ratio'] = group_turnover


        result = {}
        result['ic_decay'] = ics_decay
        result['ic_series'] = ic_series.iloc[:,0]
        result['group_nvs'] = group_nvs
        result['ann_ret'] = perfs.loc['ret_ann']
        result['SR'] = perfs.loc['SR']
        result['TO'] = perfs.loc['turnover_ratio']

        result['excess_performance'] = round(perfs,4)
        # self.factor_data = self.raw_factor_data.copy()
        return result

    def performance_summary(self,holding_period,start_date,end_date,excess_return,base,add_shift,groups,factor_list=None,universe=None):
        all_perfs = {}
        if factor_list is None:
            factor_list = self.all_factors
        for factor in factor_list:     
            group_returns,group_weights,group_turnover = self.get_group_returns(holding_period=holding_period,start_date=start_date,end_date=end_date,excess_return=excess_return,factor_list=[factor],base=base,universe=universe,add_shift=add_shift,groups=groups,return_weight=True)
            group_returns = group_returns[factor].unstack()

            group_nvs = (group_returns+1).cumprod()
            group_nvs = group_nvs/group_nvs.iloc[0]
            # group_nvs.name = 'holding_period=' + str(holding_period)

            group_turnover = group_turnover[factor].mean() / 2
            perfs = performance_indicator(group_nvs,ret_data=True,language='en',freq=self.freq)
            perfs.loc['turnover_ratio'] = group_turnover
            perfs = round(perfs,4)
            all_perfs[factor] = perfs
            # result['excess_performance'] = round(perfs,4)
            # self.factor_data = self.raw_factor_data.copy()
        return all_perfs



    def factor_ana(self,factor,ep_group,liquidity_group,**kwargs):
        """获取不同组别factor的"""
        def get_ep_liq_group_ic(factor,group_df=ep_group,**kwargs):
            ics = {}
            group_set = ['low','medium','high']#set(group_df.values)
            for group in group_set:
                ics[group] = ic_analysis(FactorEvaluation(bar_data=self.bar_data,factor_data=self.factor_data[group_df==group][factor],freq=self.freq), **kwargs)
            ics['total'] = ic_analysis(FactorEvaluation(bar_data=self.bar_data,factor_data=self.factor_data[factor],freq=self.freq), **kwargs)
            ics = pd.concat(ics)
            # ics = ics.reindex(['total'] + ['low','moderate_low','medium','moderate_high','high'],axis=1)
            return ics
        res = {}
        res['ep groups'] = get_ep_liq_group_ic(factor,**kwargs)
        # 不同ep组别ic均值输出
    
        # bar2 = px.bar(t[t.index.get_level_values(2)=='ic.mean'].droplevel(2).unstack().droplevel(0,axis=1),barmode='group',title=f'avg.ic of different ep groups of {factor}')
        # # HTML(bar2.to_html())
        # # bar2.show()
        
        # 不同liquidity组别ic均值输出
        res['liquidity groups'] = get_ep_liq_group_ic(factor,group_df=liquidity_group,**kwargs)
        # bar3 = px.bar(t[t.index.get_level_values(2)=='ic.mean'].droplevel(2).unstack().droplevel(0,axis=1),barmode='group',title=f'avg.ic of different liquidity groups of {factor}')
        # # HTML(bar3.to_html())
        # bar3.show()

        return round(pd.concat(res).unstack().droplevel(0,axis=1),4)


def summary_plot(report):

    
    fig_ic_decay = px.bar(report['ic_decay'].mean(),title='ic decay')
    fig_ic_decay.show()


    ic_series = report['ic_series']
    ic_series_ma = ic_series.rolling(12).mean()
    fig_ic_series = make_subplots()
    index = ic_series_ma.index
    fig_ic_series.add_trace(
        go.Scatter(x=index ,y=ic_series_ma.values.tolist(),name='ic_ma12')
    )
    fig_ic_series.add_trace(
        go.Bar(x=index, y=ic_series.values.tolist(), name="ic")
    )
    fig_ic_series.update_layout(title='ic series')
    fig_ic_series.show()

    group_nvs = report['group_nvs']
    fig_group_nvs = px.line(group_nvs-1)
    fig_group_nvs.update_layout(title='cumulative excess return(compound) of different groups' + '-' + group_nvs.name)
    fig_group_nvs.show()

    ann_ret = report['ann_ret']
    fig_ann_ret = px.bar(ann_ret,title='ann_ret')
    fig_ann_ret.show()


    SR = report['SR']
    fig_SR = px.bar(SR,title='SR')
    fig_SR.show()


    TO = report['TO']
    fig_TO = px.bar(TO,title='TO')
    fig_TO.show()



    perfs = report['excess_performance']
    table_perfs = create_table(perfs.reset_index())
    table_perfs.show()



    
def ic_analysis(fe,**kwargs):
    """
    fe: 自建类FactorEvaluation
    """
    def _ic_analysis(fe,**kwargs):
        ic_df = fe.get_factor_ics(method='normal',**kwargs)
        rankic_df = fe.get_factor_ics(method='rank',**kwargs)
        result_df = pd.DataFrame(columns = ic_df.columns)
        result_df.loc['ic.mean'] = ic_df.mean()
        # result_df.loc['ic.ir'] = ic_df.mean() /  ic_df.std()
        result_df.loc['ic.t-stats'] = ic_df.mean() /  ic_df.std() * ((ic_df.__len__()) ** 0.5)
        # result_df.loc['rankic.mean'] = rankic_df.mean()
        # result_df.loc['rankic.ir'] = ic_df.mean() /  ic_df.std()
        # result_df.loc['rankic.t-stats'] = ic_df.mean() /  ic_df.std() * ((ic_df.__len__()/12) ** 0.5)

        return result_df
    ic_result = {}
    return  _ic_analysis(fe,**kwargs)
    # fe.industry_neutralize()
    # ic_result['ic_ind_neu'] =  _ic_analysis(fe,**kwargs)
    # ic_result = pd.concat(ic_result)
    # return ic_result
