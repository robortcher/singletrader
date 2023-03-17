import sys
sys.path.append(r'D:\projects\singletrader_pro')
from singletrader.datasdk.qlib.base import MultiFactor
from singletrader.backtesting.bar import bar_resample
import pandas as pd
from singletrader.shared.utility import load_pkl,save_pkl
import os 
from singletrader.processors.cs_processor import CsWinzorize

cs_win = CsWinzorize(k=0.05,method='qtile-median')
data_path = r'D:\projects\singletrader_pro\samples\factortest_template\data\temp_data_add.pkl'


fields = []
names = []

fields += ['$close','$open','$high','$low','$avg','$volume','$circulating_market_cap','$turnover_ratio','$money','1/$pe_ratio']
bars = ['close','open','high','low','avg','volume','circulating_market_cap','turnover_ratio','money','ep']
names += bars

def get_data():
    global names
    global fields
    if os.path.exists(data_path):
        return load_pkl(data_path)
  
    else:
        
        mf = MultiFactor(field=fields,name=names,start_date='2009-01-01',end_date='2022-12-31')
        raw_data = mf._data
        # skew
        skew = raw_data.groupby('code').apply(lambda x:x['close'].droplevel('code').pct_change().resample('M').apply(lambda x:x.skew()))
        skew = skew.stack().swaplevel(0,1)
        skew.name = 'skew'


        # adjskew
        adjskew = cs_win(skew)
        adjskew.name = 'adjskew'


        # stddev_diff
        stddev_diff = raw_data.groupby('code').apply(lambda x:(x['close'].droplevel('code').pct_change().resample('M').apply(lambda x:x.std())).diff())
        stddev_diff = stddev_diff.stack().swaplevel(0,1)
        stddev_diff.name = 'stddev_diff'



        #年度最高价距离计算 1 - close/Max(high,252)
        distance = raw_data.groupby('code').apply(lambda x:(1-x['close'] / x['high'].rolling(252).max()).droplevel('code').resample('M').last())
        distance =  distance.stack().swaplevel(0,1)
        distance.name = 'distance'


        # 行情数据降至月频率
        bar_monthly = bar_resample(raw_data[bars],frequency='M')

        # month动量
        mom =  [bar_monthly.groupby('code').apply(lambda x:x['close']/x['close'].shift(i)-1).droplevel(0) for i in [1,2,3,6]]
        mom = pd.concat(mom,axis=1)
        mom.columns = ['mom%dM' % i for i in [1,2,3,6]]

        mom6x3 = bar_monthly.groupby('code').apply(lambda x:x['close'].shift(3)/x['close'].shift(6)-1).droplevel(0)
        mom6x3.name = 'mom6x3'


        mom9x3 = bar_monthly.groupby('code').apply(lambda x:x['close'].shift(3)/x['close'].shift(9)-1).droplevel(0)
        mom9x3.name = 'mom9x3'

        mom12x3 = bar_monthly.groupby('code').apply(lambda x:x['close'].shift(3)/x['close'].shift(12)-1).droplevel(0)
        mom12x3.name = 'mom12x3'


        #3month turnover
        turnover3M =  bar_monthly.groupby('code').apply(lambda x:x['turnover_ratio'].rolling(3).sum()).droplevel(0)
        turnover3M.name = 'turnover3M'


        #过去3 months成交额中位数
        median_turnover= bar_monthly['money'].groupby('code').apply(lambda x:x.rolling(3).median())
        median_turnover.name = 'amount3M'

        #数据合并对齐
        merged_data = pd.concat([bar_monthly, skew, adjskew,distance, mom6x3,mom9x3,mom12x3,mom,turnover3M,median_turnover,stddev_diff],axis=1)
        save_pkl(merged_data,data_path)
        return merged_data


if __name__ == '__main__':
    get_data()