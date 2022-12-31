import datetime
from datetime import timedelta
import pandas as pd
import jqdatasdk as jq
from functools import partial

current_date = datetime.datetime.now()
last_day = current_date - timedelta(1)



def get_industry_cons(start_date=None, end_date=None, trade_date=last_day, name='sw_l1'):
    ############行业成分##########
    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date

    all_trade_dates = jq.get_trade_days(
        start_date=start_date, end_date=end_date)
    if all_trade_dates.__len__() == 0:
        return pd.DataFrame()

    def __get_daily_cons(ind, date, ind_names):
        ind_name = ind_names[ind]
        ind_cons = jq.get_industry_stocks(ind, date=date)
        ind_names = [ind_name] * len(ind_cons)
        dates_index = [date] * len(ind_cons)
        df = pd.DataFrame(
            {'date': dates_index, 'code': ind_cons, 'industry_name': ind_names})
        return df

    res = pd.DataFrame()
    for date in all_trade_dates:
        universe = jq.get_all_securities(date=date).index
        all_sw_industry = jq.get_industries(name=name, date=date)
        all_sw_industry2 = all_sw_industry.reset_index()

        ind_names = all_sw_industry['name']

        func = partial(__get_daily_cons, date=date, ind_names=ind_names)
        ind_cons = all_sw_industry2.iloc[:, 0].apply(func)
        ind_cons = pd.concat(ind_cons.values).reset_index(drop=True)

        res = pd.concat([res, ind_cons])
    return res

Ind_info = get_industry_cons().set_index('code')['industry_name']