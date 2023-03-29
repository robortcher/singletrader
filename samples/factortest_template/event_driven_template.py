# get ic decay
from get_data_daily import get_data,bars
from singletrader.processors.cs_processor import CsWinzorize
from singletrader.shared.utility import save_pkl
from singletrader.factorlib import FactorEvaluation,summary_plot
import pandas as pd
import plotly.express as px
from plotly.figure_factory import create_table
import warnings
from tools.workflow import MultiFactorTesting
warnings.filterwarnings('ignore')


data = get_data().dropna() #获取处理好的数据
features = [_f for _f in data.columns if _f not in bars] + ['ep','circulating_market_cap']
bar_data = data[bars]
# bar_data.head()


liquidity_group = data.groupby(level=0).apply(lambda x:pd.qcut(x['amount60D'],3,labels=['low','medium','high'])).droplevel(0)
ep_group = data.groupby(level=0).apply(lambda x:pd.qcut(x['ep'],3,labels=['low','medium','high'])).droplevel(0)
high_liq =(liquidity_group=='high').astype(int)
print('high liquidity set\n',high_liq.head())

fe = FactorEvaluation(bar_data=data[bars],factor_data=data[features],winzorize=False,standardize=False,industry_neutralize=False)

factor = 'n_std'
report = fe.get_factor_detail_report(factor=factor,universe=high_liq,add_shift=1,base='close',start_date='2006-01-01',end_date='2022-12-31',total=False,excess_return=True,holding_period=1)
print('paused')
