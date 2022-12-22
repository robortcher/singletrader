import os 
from singletrader.backtesting.bar import bar_resample
from singletrader.backtesting.account import Account
from singletrader.performance.common import performance_indicator
from strategies import BlackLittermanAllocation
from utils.config import *
from singletrader.backtesting.engine import Engine
import numpy as np
import pandas as pd
"""
Project: Webull Investment Analysis #项目
|- Strategy Project #策略项目文件夹
    |- Robo Advisor Portfolio #策略文件夹
        |- data #数据文件夹/文件
        |- model #建模代码文件夹/文件
        |- backtest #回测代码文件夹/文件
        |- Livetrade #模拟上线代码文件夹/文件
        |- report #策略报告文件夹/文件
    |- Pre-built Portfolio
    |- Other
|- Packages #自研包文件夹，单独管理开发人员工具包
    |- Packages-developer-1
    |- Packages-developer-2
"""


w_mkt = pd.Series(1/len(asset_webull),index=asset_webull['ETF'])
target_vol = np.arange(0.05,0.17,0.01)
constraint = lambda x:x<= asset_webull.set_index('ETF').reindex(w_mkt.index)['max_weight'].values
sector_constraint = constraint_state_street
all_etf = asset_webull['ETF'].tolist()
sec_mapper = asset_webull.set_index('ETF')['asset type'].to_dict()
test_sector_mappers = {key:{'sector_mapper':sec_mapper,'sector_upper':constraint_state_street[key]['sector_upper'],'sector_lower':constraint_state_street[key]['sector_lower']} for key in constraint_state_street.keys()}
engine = Engine(universe=all_etf,start_date='2010-01-01',end_date='2022-11-30')


DATA_PATH = r'/Users/xiao/work/wealth_dev/wm-etf-allocation/example/data/ETF'
data_files = os.listdir(DATA_PATH)
local_data = {symbol.replace('_US_Equity.csv',''):pd.read_csv(DATA_PATH+'/'+ symbol) for symbol in data_files}
df = pd.concat(local_data).rename(columns={'Date':'date','OpenPrice':"open",'High':'high','Low':'low','ClosePrice':'close'}).droplevel(1)
df.index.name='symbol'
df = df[df.index.get_level_values('symbol').isin(all_etf)]
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date',append=True).swaplevel(0,1)
df_month = bar_resample(df,'MS')



def run(engine):
    for sec_mapper_config in test_sector_mappers:
        account = Account(
            capital_base = 1e8,
            commission_buy = 0.000,
            commission_sell = 0.000,
            margin_ratio = 1,
            min_commission = 0.,
            trade_free = False,
            price_type = 'close',
            slippage = 0.,
            slippage_ratio = 0.,
            tax_buy = 0.,
            tax_sell = 0.00,
            region='us',
        )
        mapper = test_sector_mappers[sec_mapper_config]
        engine.run_backtest(expanding=True,
                            strategy=BlackLittermanAllocation,
                            constraint=constraint,
                            sector_constraint=mapper,
                            has_sector_constraint=True,
                            am_size=70,
                            account=account,
                            w_mkt=w_mkt)
    return engine.get_all_nv()
engine.load_data(src='yahoo', interval="1mo")
nv_yahoo = run(engine)
engine.get_all_info('net_value')
print('=')