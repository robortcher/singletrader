# -*-coding:utf-8 -*-
"""
从jq获取数据到pg数据库
"""
from jqdatasdk import auth, logout
from jqdatasdk import finance, get_index_weights,query, get_fundamentals, valuation, income, get_all_securities, get_trade_days, indicator,get_price,get_call_auction
import pandas as pd
import datetime
import jqdatasdk as jq
try:
    from pgsql_common import Postgres
except:
    from .pgsql_common import Postgres
import os
import json
from datetime import timedelta
from .config import ValuationConfigPG,PricePostConfigPG,AuctionConfigPG

import logging

class DataSaver():
    def __init__(self,start_date=None,end_date=None,trade_date=None,mode='all',func=None):
        
        assert mode in ('all','split'), logging.error(f"不支持的下载模式{mode},请选择split或者all")
        
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
        start_date_1 = start_date#[:4] + '-01-01'
        end_date_1 = end_date#[:4] + '-12-31'
        start_dates = pd.date_range(start_date_1,end_date_1,freq=freq)  
        end_dates = [i-timedelta(1) for i in start_dates[1:]]
        start_dates = [i.strftime('%Y-%m-%d') for i in start_dates]
        end_dates = [i.strftime('%Y-%m-%d') for i in end_dates]
        end_dates.append(end_date_1)
        periods = list(zip(start_dates, end_dates))
        return periods
    
    def data_writer(self,**kwargs):
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
        


def download_valuation(pg=None, start_date='2005-01-01',end_date=None,trade_date=None,is_daily=False):
    """
    daily 更新前一天的数据
    """
    st = get_all_securities(['stock'])
    stocks = list(st.index)
    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date
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
        if pg is not None:
            pg.update_insert_df(df0, f"valuation",
                                text_columns,  # text
                                constraint_columns,  # constraint
                                date_columns=date_columns,
                                )
            print(f'inserted {day}')
        else:
            print(df0)

# 复权行情数据
def down_price(pg=None,start_date='2005-01-01',end_date=None,trade_date=None,**kwargs):
    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date
    universe = get_all_securities().index.tolist()
    factor_list =  ['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor']
    df = get_price(universe, start_date=start_date, end_date=end_date, frequency='daily', fields=factor_list, skip_paused=False, fq='post').rename(columns = {'time':'date'})
    text_columns = ['code']
    date_columns = ['date']
    constraint_columns = ['date', 'code']
    if pg is not None:
        pg.update_insert_df(df, f"price_post",
                        text_columns,  # text
                        constraint_columns,  # constraint
                        date_columns=date_columns,
                        )
    else:
        return df



def down_price_minute(pg=None,start_date='2005-01-01',end_date=None,trade_date=None,**kwargs):
    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date
    universe = get_all_securities().index.tolist()
    days = get_trade_days(start_date=start_date,
                        end_date=end_date, count=None)
    for day in days:
        start_date = day.strftime('%Y-%m-%d')+ ' ' + '00:00:00'
        end_date = day.strftime('%Y-%m-%d')+ ' ' + '15:00:00'
        factor_list =  ['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor']
        df = get_price(universe, start_date=start_date, end_date=end_date, frequency='1m', fields=factor_list, skip_paused=False, fq='post').rename(columns = {'time':'date'})
        text_columns = ['code']
        timestamp_columns = ['date']
        constraint_columns = ['date', 'code']
        if pg is not None:
            pg.update_insert_df(df, pg.conf.table_name,
                            text_columns,  # text
                            constraint_columns,  # constraint
                            timestamp_columns=timestamp_columns,
                            )
        else:
            return df



# 集合竞价
def download_auction(pg=None,start_date='2005-01-01',end_date=None,trade_date=None,**kwargs):
    universe = get_all_securities().index.tolist()
    df = get_call_auction(security=universe, start_date=start_date, end_date=end_date).rename(columns={'time':'date','volume':'volume_au',"money":'money_au'})
    df['date'] = pd.to_datetime(df['date'].apply(lambda x:x.strftime('%Y-%m-%d')))
    # df = df['code', 'time', 'current', 'volume', 'money', 'a1_p', 'a1_v', 'a2_p', 'a2_v', 'a3_p', 'a3_v', 'a4_p', 'a4_v', 'a5_p', 'a5_v', 'b1_p', 'b1_v', 'b2_p', 'b2_v', 'b3_p', 'b3_v', 'b4_p', 'b4_v', 'b5_p', 'b5_v']
    text_columns = ['code']
    date_columns = ['date']
    constraint_columns = ['date', 'code']
    if pg is not None:
        pg.update_insert_df(df, f"auction",
                            text_columns,  # text
                            constraint_columns,  # constraint
                            date_columns=date_columns,
                            )
    else:
        return df



#成 分股权重
def download_index_cons(pg=None,start_date=None,end_date=None):

    days = get_trade_days(start_date=start_date,
                        end_date=end_date, count=None)
    
    for day in days:
        df = get_index_weights(index_id='000300.XSHG',date=day)
        df.index.name = 'code'
        df = df.reset_index()
        text_columns = ['code','display_name']
        date_columns = ['date']
        constraint_columns = ['date', 'code']
        if pg is not None:
            pg.update_insert_df(df, pg.conf.table_name,
                                text_columns,  # text
                                constraint_columns,  # constraint
                                date_columns=date_columns,
                                )
        else:
            return df


def download_index_price(pg=None,start_date=None,end_date=None):
    index_list = ['000300.XSHG','000016.XSHG','000905.XSHG','000852.XSHG']
    df =get_price('000300.XSHG', start_date= start_date,end_date=end_date, frequency='daily', fields=['open','close','low','high','volume','money','high_limit','low_limit','avg','pre_close'])
    pass



def AllWriter(start_date='20005-01-01',end_date=datetime.datetime.now().strftime('%Y-%m-%d')):
    pg_v = Postgres(conf=ValuationConfigPG)
    vw = DataSaver(start_date=start_date,end_date=end_date,func=download_valuation)
    vw.data_writer(pg=pg_v)

    pg_w= Postgres(conf=PricePostConfigPG)
    pw = DataSaver(start_date=start_date,end_date=end_date,func=down_price,mode='split')
    pw.data_writer(pg=pg_w)
    
    

    pg_au = Postgres(conf=AuctionConfigPG)
    auw =  DataSaver(start_date=start_date,end_date=end_date,func=download_auction,mode='split')
    auw.data_writer(pg=pg_au)
    pass 


def UpdateWriter(start_date=None,end_date=None,trade_date=None):
    if trade_date is not None:
        start_date = trade_date 
        end_date = trade_date 
    
    pg_v = Postgres(conf=ValuationConfigPG)
    vw = DataSaver(start_date=start_date,end_date=end_date,func=download_valuation)
    vw.data_writer(pg=pg_v)

    pg_w= Postgres(conf=PricePostConfigPG)
    pw = DataSaver(start_date=start_date,end_date=end_date,func=down_price)
    pw.data_writer(pg=pg_w)
    
    

    pg_au = Postgres(conf=AuctionConfigPG)
    auw =  DataSaver(start_date=start_date,end_date=end_date,func=download_auction)
    auw.data_writer(pg=pg_au)


if __name__ == '__main__':
    import sys



    

