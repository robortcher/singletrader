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
from singletrader.processors.cs_processor import IndAggregation,CsWinzorize,CsStandardize
from singletrader.constant import Ind_info
import pandas as pd




data = get_data()

# 获取因子列表，剔除行情数据
features = [_f for _f in data.columns if _f not in bars] + ['ep']#['circulating_market_cap']
features_with_mv = features + ['circulating_market_cap']


cs_win = CsWinzorize()
cs_std = CsStandardize()
ind_agg = IndAggregation() #创建IndAggregation对象（自建类用于行业聚合处理）

# mom1M收益合成行业价格序列（cap）
return1m_ind_agg_cap = ind_agg(data[['mom1M','circulating_market_cap']],weights='circulating_market_cap',ind_info=Ind_info)
price_ind_cap_cap = (1+return1m_ind_agg_cap['mom1M'].unstack()).cumprod().stack()



# mom1M收益合成行业价格序列（cap）
return1m_ind_agg_eq = ind_agg(data[['mom1M']],weights='eq',ind_info=Ind_info)
price_ind_cap_eq = (1+return1m_ind_agg_eq['mom1M'].unstack()).cumprod().stack()


data[features] = cs_win(data[features])
data[features] = cs_std(data[features])

# 等权聚合
ind_agg_data_eq = ind_agg(data,weights='eq',ind_info=Ind_info)
# mom1M收益合成行业价格序列（eq）
price_ind_eq = (1+ind_agg_data_eq['mom1M'].unstack()).cumprod().stack()






