import sys
sys.path.append(r'D:/projects/singletrader/')
# import singletrader
# from singletrader.datautils.qlibapi.constructor.base import MultiFactor
import pandas as pd
import warnings
from singletrader.factortesting.factor_testing import FactorEvaluation
from pprint import pprint
import numpy as np
warnings.filterwarnings('ignore')
import plotly.express as px




#合并数据示例
from pathlib import Path
from workflow import MultiFactorTesting
__file__ = r'D:\projects\singletrader\samples\double_sort_framework\workflow.ipynb'
file = Path(__file__)
parent_path = file.parent
data_file = parent_path.__str__() + '/data/' + r'price_and_factor_data.csv'
merged_data = pd.read_csv(data_file)
merged_data['date'] = pd.to_datetime(merged_data['date']) #date字段需要datetime格式；
merged_data = merged_data.set_index(['date','asset']) #必须为date,asset双重索引；
merged_data = merged_data.dropna()
merged_data.dropna().head()

fields = []
names = []

features = ['skew',	'distance',	'mom3M', 'turnover3M']
fields += ['$close','$open','$high','$low','$avg','$volume','$circulating_market_cap','$turnover_ratio']
bars = ['close','open','high','low','avg','volume','market_cap','turnover_ratio']
names += bars



fe = FactorEvaluation(merged_data[bars],merged_data[features],freq=12)

normal_ic = fe.get_factor_ic(base='close-open',add_shift=0,method='normal')
rank_ic = fe.get_factor_ic(base='close-open',add_shift=0,method='rank')
res_ic = pd.DataFrame(columns=normal_ic.columns)


res_ic.loc['ic.avg'] = normal_ic.mean()
res_ic.loc['ic.std'] = normal_ic.std()
res_ic.loc['icir'] = normal_ic.mean() / normal_ic.std()
res_ic.loc['rankic'] = rank_ic.mean()



from singletrader.performance.common import performance_indicator
r = fe.get_group_returns(add_shift=0,groups=5,return_weight=True,base='close-open')



mkt_return=merged_data.dropna().groupby(level=0).apply(lambda x:(x['close']/x['open']-1).mean()).shift(-1)


for factor in r[0].columns:
    print('different group return of '+factor,'\n')
    _per = performance_indicator((1+r[0][factor].unstack()).cumprod(),freq=12,language='en',mkt_return=mkt_return)
    _per.loc['turnover per trade'] = r[1][factor].groupby(level=0).apply(lambda x:x.diff().abs().sum(axis=1)/2).droplevel(0).unstack().T.mean()
    pprint(round(_per,2))
    print('\n')