# get ic decay
from get_data import get_data,bars
from singletrader.processors.cs_processor import CsWinzorize
from singletrader.shared.utility import save_pkl
from singletrader.factorlib import FactorEvaluation,summary_plot
import pandas as pd
import plotly.express as px
from plotly.figure_factory import create_table
import warnings
warnings.filterwarnings('ignore')
from singletrader.processors.cs_processor import IndAggregation
from singletrader.constant import Ind_info
import pandas as pd




data = get_data().dropna()


features = [_f for _f in data.columns if _f not in bars] + ['ep','circulating_market_cap']

bar_data = data[bars]
bar_data.head()


liquidity_group = data.groupby(level=0).apply(lambda x:pd.qcut(x['amount3M'],3,labels=['low','medium','high'])).droplevel(0)
ep_group = data.groupby(level=0).apply(lambda x:pd.qcut(x['ep'],3,labels=['low','medium','high'])).droplevel(0)
high_liq =(liquidity_group=='high').astype(int)
# data['adjskew'] = cs_win(data['skew'])
fe = FactorEvaluation(bar_data=data[bars],factor_data=data[features],freq=12,winzorize=True,standardize=True)

#未来一个月
ic_summary = fe.get_summary(add_shift=0,start_date='2020-01-01',universe=high_liq,end_date='2022-12-31',base='close')
round(ic_summary,4)
# fe.get_group_returns(return_weight=True,start_date='2022-01',end_date='2022-12-31')

# ic_summary= fe.get_summary(add_shift=0,base='close')


# ic_summary2 = fe.get_summary(add_shift=0,base='close',next_n=3)



# ic_summary= fe.get_ic_summary(add_shift=0,base='close')
# ic_summary2 = fe.get_ic_summary(add_shift=0,base='close',next_n=3)

factor='mom3M_GN'
report = fe.get_factor_detail_report(factor=factor,add_shift=0,base='close',start_date='2022-01-01',end_date='2022-12-31',total=False,excess_return=True,holding_period=1)
summary_plot(report)