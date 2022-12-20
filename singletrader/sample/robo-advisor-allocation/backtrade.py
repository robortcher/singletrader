import os 
from singletrader.backtesting.bar import bar_resample
from singletrader.backtesting.account import Account
from singletrader.performance.common import performance_indicator
from strategies import BlackLittermanAllocation
from utils.config import *
from singletrader.backtesting.engine import Engine
import numpy as np
import pandas as pd

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
engine.load_data(df_month.reset_index()) #本地load_data
nv_local = run(engine)
ac = engine._accounts[0]



engine.load_data(src='yahoo', interval="1mo")
nv_yahoo = run(engine)
print('=')