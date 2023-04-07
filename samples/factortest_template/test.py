"""qlib instruments生成脚本，包括sz50、hs300、zz500、zz1000、zz1800"""
import pandas as pd
import numpy as np
from singletrader.datasdk.sql.dataapi import get_index_cons
from singletrader import QLIB_BIN_DATA_PATH
index_cons = get_index_cons(start_date='2005-01-01')

# dir = r'D:\database\.singletrader\qlib_data\instruments'
dir = QLIB_BIN_DATA_PATH + r'/' + 'instruments' 


def get_start_end_date(test_data):
    test_data = test_data.droplevel('code')
    test_data_df = pd.DataFrame(test_data)
    test_data_df['group'] = np.where(test_data>0,0,1)
    test_data_df['group'] = test_data_df['group'].cumsum()
    res = test_data_df[test_data_df.iloc[:,0]>0]
    res = res.groupby('group').apply(lambda x:pd.Series([x.index[0].strftime('%Y-%m-%d'),x.index[-1].strftime('%Y-%m-%d')]))
    return res

for index in index_cons.columns:
    sz50 = index_cons[index]
    sz50 = sz50.dropna().unstack().fillna(0).stack()
    res = sz50.groupby(level=1).apply(lambda x:get_start_end_date(x)).droplevel(1)
    path = dir + '/'+index+'.txt'
    res.columns = ['start_date','end_date']
    res.to_csv(path,header=None,sep='\t')


if __name__ == '__main__':
    print("test......")