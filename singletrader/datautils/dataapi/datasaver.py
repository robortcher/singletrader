# -*-coding:utf-8 -*-
"""
从jq获取数据到pg数据库
"""
from jqdatasdk import auth, logout
from jqdatasdk import finance, query, get_fundamentals, valuation, income, get_all_securities, get_trade_days, indicator
import pandas as pd
import datetime
from datacenter_utils.pgsql_common import Postgres
import os
import json


def get_valuation(pg, is_daily=False):
    """
    daily 更新前一天的数据
    """
    st = get_all_securities(['stock'])
    stocks = list(st.index)
    if is_daily:
        days = get_trade_days(end_date=datetime.date.today(), count=2)[0:1]
    else:
        days = get_trade_days(start_date='2021-10-27',
                              end_date='2022-05-04', count=None)
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
        pg.update_insert_df(df0, f"cn_jq_stk_valuation",
                            text_columns,  # text
                            constraint_columns,  # constraint
                            date_columns=date_columns,
                            )
        print(f'inserted {day}')


if __name__ == '__main__':
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
