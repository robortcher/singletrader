from get_data import get_data,bars
from singletrader.processors.cs_processor import CsWinzorize
from singletrader.shared.utility import save_pkl

from singletrader.factorlib import FactorEvaluation
import pandas as pd

cs_win = CsWinzorize(k=0.05,method='qtile-median')





data = get_data().dropna()
features = [_f for _f in data.columns if _f not in bars]


liquidity_group = data.groupby(level=0).apply(lambda x:pd.qcut(x['amount3M'],3,labels=['low','medium','high'])).droplevel(0)
ep_group = data.groupby(level=0).apply(lambda x:pd.qcut(x['ep'],3,labels=['low','medium','high'])).droplevel(0)


liquidity_group = data.groupby(level=0).apply(lambda x:pd.qcut(x['amount3M'],3,labels=['low','medium','high'])).droplevel(0)
ep_group = data.groupby(level=0).apply(lambda x:pd.qcut(x['ep'],3,labels=['low','medium','high'])).droplevel(0)
# data['adjskew'] = cs_win(data['skew'])


fe = FactorEvaluation(bar_data=data[bars],factor_data=data[features],freq=12)
# fe.factor_ana(factor='mom3M',ep_group=ep_group,liquidity_group=liquidity_group)
report = fe.get_factor_detail_report(factor='mom3M',add_shift=0,base='close',total=False,excess_return=True)




print('===')





