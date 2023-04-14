import sys 
sys.path.append(r'D:/projects/singletrader_pro')
from singletrader.datasdk.qlib.base import MultiFactor,__bar_fields__,__bar_names__
from singletrader.factorlib.factortesting import FactorEvaluation
from singletrader.datasdk.sql.dataapi import get_index_cons
from singletrader.performance.common import performance_indicator 

if __name__ == '__main__':
    fields = ['Sum($money,60)'] + __bar_fields__ + ['$open/Ref($close,1)-1'] 
    names = ['money60']+__bar_names__ + ['Gap']
    mf = MultiFactor(field=fields,name=names,start_date='2023-01-01',end_date='2023-04-10')
    d = mf._data
    import pandas as pd
    from singletrader.shared.utility import load_pkl,save_pkl
    """

    to / yc
    
    tmo / to
    
    """
    ind_cons = get_index_cons()
    

    fe = FactorEvaluation(d.iloc[:,:-1],d.iloc[:,-1],winzorize=True,standardize=True,industry_neutralize=True)
    mm = fe.factor_data.iloc[:,-1]
    label = mm.groupby(level=0).apply(lambda x:pd.qcut(x.rank(method='first'),10,labels=(0,1,2,3,4,5,6,7,8,9)))
    save_pkl(mm[label>0],r'D:\projects\singletrader_pro\samples\contribute\trademodel\lgbmodel\predict\gap_signal.pkl')
    ic = fe.get_factor_ic(add_shift=0,base='close-open')
    fe.get_factor_ics(periods=(1,2,5,10),add_shift=0,base='open')
    print('=====')