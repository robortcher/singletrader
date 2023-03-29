from singletrader.datasdk.qlib.base import MultiFactor
from singletrader.shared.utility import load_pkls,load_pkl
from singletrader.factorlib.factortesting import FactorEvaluation
from singletrader.performance.common import performance_indicator
from configs import basic_config
from singletrader.constant import CONST
import pandas as pd
import jqdatasdk as jq
import numpy as np
import datetime
from singletrader.factorlib.functions import industry_neutralize

import os

jq_user = os.environ.get('JQ_USER')
jq_passwd = os.environ.get('JQ_PASSWD')
jq.auth(jq_user,jq_passwd)

time = datetime.datetime.now().strftime('%H:%M:%S')

fields = ['$close','$open','$high','$low','$avg','$volume','$market_cap']
bars = ['close','open','high','low','avg','volume','mv']
names = bars


if __name__ == '__main__':
    field,name = basic_config()
    
    if time >= '18:00:00':
        last_trade_date = CONST.CURRENT_TRADE_DAY
    else:
        last_trade_date = CONST.LAST_TRADE_DAY
    


    model =load_pkl(r'D:\projects\singletrader_pro\samples\contribute\trademodel\models\2023.pkl')
    mf = MultiFactor(field=field,name=name,start_date=last_trade_date,end_date=last_trade_date)
    data=mf._data
    
    pred = pd.Series(model.model.predict(data),index=data.index).droplevel(0)
  
    pred = industry_neutralize(pred)
    all_security_info = jq.get_all_securities()
    is_st = all_security_info['display_name'].apply(lambda x:'ST' in x).astype(np.int)
    pred = pred[is_st==0]

    target_cons = pred.sort_values(ascending=False).index.tolist()    

     
    target_cons = [i for i in target_cons if (not i.startswith('688') and (not i.startswith('3')))][:10]
    tgt = all_security_info.reindex(target_cons)
    
    
    mf = MultiFactor(field=fields,name=names,start_date='2018-01-01',end_date='2023-03-31')
    data=mf._data    
    
    
    
