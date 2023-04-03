from singletrader.datasdk.qlib.base import MultiFactor
from singletrader.shared.logging import logger
from singletrader.processors.cs_processor import _add_cs_data,CsWinzorize,CsStandardize,CsNeutrualize
from singletrader.constant import get_industry_cons
import pandas as pd
from singletrader.processors.operator import get_beta
from singletrader import __date_col__,__symbol_col__
cs_win = CsWinzorize(k=0.01,method='qtile')
cs_std = CsStandardize('z-score')
cs_neu = CsNeutrualize()

class Attributor():
    
    factor_fields = ["$close/Ref($close,5)-1",# 5日动量
                     "Sum($turnover_ratio,5)",# 5日换手
                     "Log($circulating_market_cap)",# 流通市值
                     "Std($close/Ref($close,1), 20)",# 月度换手率
                     "1/$pe_ratio",# EP
                     ]
    factor_names = ["mom5","turnover_ratio5","logcap","volatility","ep"]

    forward_return_fields = ["Ref($close,-1)/$close-1"]
    forward_return_names = ["daily_return"]


    def __init__(self,start_date=None, end_date=None,add_ind=True):
        self.start_date =  start_date
        self.end_date = end_date
        self.add_ind = add_ind
        self.setup_data()
        print('===')


    def setup_data(self):
        """数据装载"""
        mf = MultiFactor(start_date=self.start_date,end_date=self.end_date,field=self.forward_return_fields+self.factor_fields,name=self.forward_return_names+self.factor_names)
        data = mf._data
        data[self.factor_names] = cs_win(data[self.factor_names])
        data[self.factor_names] = cs_std(data[self.factor_names])
        if self.add_ind:
            ind_info = pd.get_dummies(get_industry_cons())
            data = _add_cs_data(data,ind_info)
        self.data = data
        
    def calculate_factor_return(self,value='params'):
        result = {}
        factor_return = self.data.groupby(level=__date_col__).apply(lambda x:get_beta(x,value=value)).shift().fillna(0)
        r2 = self.data.groupby(level=__date_col__).apply(lambda x:get_beta(x,value='rsquared_adj')).shift()
        
        result['return series'] = factor_return
        result['r2'] = r2
        return result
    
    def get_summary_report(self):
        import plotly.express as px
        result = self.calculate_factor_return()
        return_series = result['return series']
        r2 = result['r2']

        fig_series = px.line((1+return_series).cumprod(),title='return series')
        

        bar_return = px.bar(return_series.sum(),title='total_return')
        fig_series.show()
        bar_return.show()

        return result


if __name__ == '__main__':
    logger.info('test begin')


    attr = Attributor(start_date='2023-03-01',end_date='2023-03-31',add_ind=True)
    attr.setup_data()
    cs_neu(attr.data['daily_return'])
    r = attr.get_summary_report()

    logger.info('test end')