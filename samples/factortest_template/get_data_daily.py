import sys
sys.path.append(r'D:\projects\singletrader_pro')
from singletrader.datasdk.qlib.base import MultiFactor
from singletrader.backtesting.bar import bar_resample
import pandas as pd
from singletrader.shared.utility import load_pkl,save_pkl
import os 
from singletrader.processors.cs_processor import CsWinzorize
import numpy as np

cs_win = CsWinzorize(k=0.05,method='qtile-median')
# data_path = r'D:\projects\singletrader_pro\samples\factortest_template\data\temp_data_from05.pkl'
data_path = r'D:\projects\singletrader_pro\samples\factortest_template\data\temp_data_from05_daily.pkl'


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
        # 价格偏移n-std
        fields += ['($close - Sum($close,250)/250) / Std($close,250)']
        names += ['n_std']

        fields += ['Med($money,60)']
        names += ['amount60D']



        mf = MultiFactor(field=fields,name=names,start_date='2005-01-01',end_date='2022-12-31')
        raw_data = mf._data
        raw_data['excess_3std'] = np.where(raw_data['n_std']>=3,1,0)
        raw_data['excess_3std'] = np.where(raw_data['n_std']<=-3,-1,raw_data['excess_3std'])
        save_pkl(raw_data[raw_data.index.get_level_values(0)>='2006'],data_path)
        
if __name__ == '__main__':
    get_data()