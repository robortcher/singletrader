from jqdatasdk import get_trade_days,auth
import datetime
from datetime import timedelta
import pandas as pd
from functools import partial
import jqdatasdk as jq
import os
auth(os.environ.get('JQ_USER') ,
    os.environ.get('JQ_PASSWD'))
class CONST():
    LAST_DAY =  (datetime.datetime.now()-timedelta(1)).strftime('%Y-%m-%d')
    CURRENT_DAY = datetime.datetime.now().strftime('%Y-%m-%d')
    LAST_TRADE_DAY = get_trade_days(count=1,end_date = LAST_DAY)[-1].strftime('%Y-%m-%d')
    CURRENT_TRADE_DAY = get_trade_days(count=1,end_date = CURRENT_DAY)[-1].strftime('%Y-%m-%d')
    IS_TRADE_DATE = 1 if CURRENT_DAY == CURRENT_TRADE_DAY  else 0
    IS_TRADE_DATE_LAST_DAY = 1 if LAST_DAY  ==  LAST_TRADE_DAY else 0
