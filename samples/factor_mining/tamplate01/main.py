# get ic decay
from singletrader.processors.cs_processor import CsWinzorize,CsNeutrualize
from singletrader.shared.utility import save_pkl
from singletrader.factorlib import FactorEvaluation,summary_plot
import pandas as pd
import plotly.express as px
from plotly.figure_factory import create_table
import warnings
warnings.filterwarnings('ignore')
from singletrader.datasdk.qlib.base import MultiFactor
from singletrader.shared.utility import load_pkls
from singletrader.datasdk.sql.dataapi import get_index_cons
from singletrader.factorlib.factortesting import summary_plot
cs_neu = CsNeutrualize()


from config import get_feature_configs

field_features,name_features = get_feature_configs()

field_bar = ['$close','$avg','$open','$low','$volume','Sum(1-$paused,20)']
name_bar = ['close','avg','open','low','volume','used']

fields = field_bar + field_features
names = name_bar+name_features

index_cons = get_index_cons(start_date='2015-01-01',end_date='2023-03-31')
if __name__ == '__main__':
    mf = MultiFactor(field=fields,name=names,start_date='2015-01-01',end_date='2023-03-31')
    data = mf._data
    fe = FactorEvaluation(bar_data=data[name_bar],factor_data=data[name_features])
    summary = fe.get_summary(holding_period=1,base='avg',excess_return=True,total=False)
    ic = fe.get_factor_ic(base='avg',next_n=1,method='rank')
    ic_normal = fe.get_factor_ic(base='avg',next_n=2,method='normal')



    print('===')
