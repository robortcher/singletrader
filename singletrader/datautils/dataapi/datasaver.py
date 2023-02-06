# -*-coding:utf-8 -*-
"""
从jq获取数据到pg数据库
"""
from jqdatasdk import auth, logout
from jqdatasdk import finance, query, get_fundamentals, valuation, income, get_all_securities, get_trade_days, indicator
import pandas as pd
import datetime
from pgsql_common import Postgres
import os
import json
from datetime import timedelta
from config import ValuationConfigPG


class DataRetrive():
    def __init__(self,start_date=None,end_date=None,trade_date=None,mode='all',func=None):
        if trade_date is not None:
            mode='all'
            start_date=trade_date
            end_date=trade_date
        
        self.start_date=start_date
        self.end_date=end_date
        self.trade_date=trade_date
        self.mode=mode
        self.func=func
    
    def date_split(self,**kwargs):
        start_date=kwargs.get('start_date',self.start_date)
        end_date=kwargs.get('end_date',self.end_date)
        # trade_date=kwargs.get('trade_date',self.trade_date)
        freq=kwargs.get('freq',"YS")
        start_date_1 = start_date[:4] + '-01-01'
        end_date_1 = end_date[:4] + '-12-31'
        start_dates = pd.date_range(start_date_1,end_date_1,freq=freq)  
        end_dates = [i-timedelta(1) for i in start_dates[1:]]
        start_dates = [i.strftime('%Y-%m-%d') for i in start_dates]
        end_dates = [i.strftime('%Y-%m-%d') for i in end_dates]
        end_dates.append(end_date_1)
        periods = list(zip(start_dates, end_dates))
        return periods
    
    def datawriter(self,**kwargs):
        start_date=kwargs.get('start_date',self.start_date)
        end_date=kwargs.get('end_date',self.end_date)
        trade_date=kwargs.get('trade_date',self.trade_date)
        if self.mode == 'split':
            periods = self.date_split(**kwargs)
            for _period in periods:
                self.func(start_date=_period[0],end_date=_period[1],**kwargs)        
        elif self.mode == 'all':
            self.func(start_date=start_date,end_date=end_date,trade_date=trade_date,**kwargs)
    
    # def download(self,start_date=None,end_date=None,trade_date=None):
    #     """下载数据的函数"""
    #     pass
        


def get_valuation(pg, start_date='2005-01-01',end_date=None,trade_date=None,is_daily=False):
    """
    daily 更新前一天的数据
    """
    st = get_all_securities(['stock'])
    stocks = list(st.index)
    if is_daily:
        days = get_trade_days(end_date=datetime.date.today(), count=2)[0:1]
    else:
        days = get_trade_days(start_date=start_date,
                              end_date=end_date, count=None)
    for day in days:
        df = get_fundamentals(query(
            valuation
        ).filter(
            valuation.code.in_(stocks)
        ), date=day)

        if df.empty:
            print(f'{day} empty')
            continue
        del df['id']
        df.rename(columns={'day': 'date'}, inplace=True)
        text_columns = ['code']
        date_columns = ['date']
        constraint_columns = ['date', 'code']
        columns = list(df.columns)
        columns.insert(0, columns.pop(columns.index('code')))
        columns.insert(0, columns.pop(columns.index('date')))
        df0 = df[columns]
        pg.update_insert_df(df0, f"valuation",
                            text_columns,  # text
                            constraint_columns,  # constraint
                            date_columns=date_columns,
                            )
        print(f'inserted {day}')


if __name__ == '__main__':
    
    dw = DataRetrive(start_date='2005-01-01',end_date='2023-02-05',mode='split',func=get_valuation)
    p = Postgres(conf=ValuationConfigPG)
    
    dw.datawriter(pg=p)
    
    jq_name = str(os.getenv('JQ_NAME_MAIN'))
    jq_passwd = str(os.getenv('JQ_PASSWORD_MAIN'))
    auth(jq_name, jq_passwd)
    conf_str = os.getenv('PG_CONF_STATIC')
    if conf_str:
        config = json.loads(conf_str)
    p = Postgres(conf=config)
    # # get_valuation(p)
    get_valuation(p, is_daily=True)

    logout()
