
import pandas as pd
import datetime
import jqdatasdk as jq
from functools import partial
import pandas as pd
import datetime
import os


INDEX_CODES = ['000016.XSHG', '000300.XSHG', '000905.XSHG', '000852.XSHG', '000688.XSHG', '399001.XSHE',
               '399006.XSHE', '000001.XSHG']

INDEX_NAMES = {
    '000016.XSHG': 'sz50',
    '000300.XSHG': 'hs300',
    '000905.XSHG': 'zz500',
    '000852.XSHG': 'zz1000',
    '000688.XSHG': 'kc50',
    '399001.XSHE': 'szcz',
    '399006.XSHE': 'cybz',
    '000001.XSHG': 'szzs'
}
jq.auth(os.environ['JQ_USER'],os.environ['JQ_PASSWD'])


get_all_securities = jq.get_all_securities
get_trade_days = jq.get_trade_days


def get_security_info(**kwargs):
    instruments = get_all_securities()
    instruments.index.name = 'code'
    instruments = instruments.reset_index()
    return instruments

def get_industry_cons(start_date=None, end_date=None, trade_date=None, name='sw_l1'):
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


def get_caps(start_date=None, end_date=None, trade_date=None):
    ###########流通市值#########

    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date
    daterange = jq.get_trade_days(start_date=start_date, end_date=end_date)

    q = jq.query(jq.valuation.capitalization,
                 jq.valuation.circulating_cap,
                 jq.valuation.code,
                 jq.valuation.day)

    def get_data(q, x):
        df = jq.get_fundamentals(q, date=x)
        return df
    dflist = list(map(lambda x: get_data(q, x), daterange))
    # dflist = list(filter(lambda x: not x.empty, dflist))
    # if dflist:
    data = pd.concat(dflist).rename(columns={'day': 'date'})
    # else:
    #     raise Exception('get_caps failed')
    # data['date'] = pd.to_datetime(data['date'])
    data['date'] = data['date'].apply(lambda x: pd.to_datetime(x).date())
    return data


def get_eps(start_date=None, end_date=None, trade_date=None):
    ##########eps########

    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date
    stocks = jq.get_all_securities(
        types=['stock'], date=trade_date).index.to_list()
    df = jq.get_factor_values(stocks, 'eps_ttm', start_date, end_date)[
        'eps_ttm'].T
    value_vars = df.columns
    df['code'] = df.index
    df_res = pd.melt(df, id_vars=[
                     'code'], value_vars=value_vars, var_name='date', value_name='eps_ttm')
    df_res['date'] = df_res['date'].apply(lambda x: x.date())
    return df_res[['code', 'date', 'eps_ttm']]


def get_index_weights(index_list=INDEX_CODES, start_date=None, end_date=None, trade_date=None):
    ###########指数成分##############
    """单位为%"""
    if trade_date is not None:
        start_date = jq.get_trade_days(start_date=trade_date)[1]
        end_date = jq.get_trade_days(start_date=trade_date)[1]
    index_name_list = [INDEX_NAMES[ind] for ind in INDEX_CODES]
    all_trade_dates = jq.get_trade_days(
        start_date=start_date, end_date=end_date)
    start_month_firstday = datetime.datetime(
        all_trade_dates[0].year, all_trade_dates[0].month, 1)
    end_month_firstday = datetime.datetime(
        all_trade_dates[0].year, all_trade_dates[-1].month, 1)
    month_starts = pd.date_range(
        start_month_firstday, end_month_firstday, freq='MS')
    month_starts = [i.date() for i in month_starts]
    new_all_trade_dates = sorted(
        list(set(all_trade_dates.tolist() + month_starts)))
    datas = []
    # print(f'all_trade_dates:{all_trade_dates} \n start_month_firstday:{start_month_firstday} \n end_month_firstday:{end_month_firstday}\n ')
    # print(f'month_starts:{month_starts}\n new_all_trade_dates:{new_all_trade_dates}')
    def __get_index_weights(index_code, all_trade_dates, month_starts):
        datas = []
        for date in month_starts:
            universe = jq.get_all_securities(date=date).index
            daily_cons = jq.get_index_weights(
                index_code, date=date)[['weight']]
            daily_cons = daily_cons.reindex(universe).fillna(0)
            daily_cons.index.name = 'code'
            daily_cons['date'] = date
            datas.append(daily_cons)
        datas = pd.concat(datas).reset_index()
        datas = datas.set_index(['date', 'code'])
        return datas
    for index in index_list:
        datas.append(__get_index_weights(
            index, all_trade_dates=all_trade_dates, month_starts=month_starts))
    datas = pd.concat(datas, axis=1)
    datas = datas.groupby(level=1).apply(lambda x: x.droplevel(1).reindex(
        new_all_trade_dates).ffill().reindex(all_trade_dates)).swaplevel(0, 1)
    datas.columns = index_name_list
    datas = datas.reset_index()
    # print(datas)
    if trade_date is not None:
        datas['date'] = trade_date
    return datas


def get_summary_data(start_date=None, end_date=None, trade_date=None, index_col=['date', 'code'], func_list=[get_industry_cons, get_caps, get_eps, get_index_weights]):
    #########获取汇总数据#############

    if trade_date is not None:
        start_date = trade_date
        end_date = trade_date
    datas = []
    for func in func_list:
        data = func(start_date=start_date, end_date=end_date,
                    trade_date=trade_date)
        # if not data.empty:
        # print(str(func))
        # print(data)
        # print(data['date'].dtype)
        data = data.set_index(index_col)
        datas.append(data)
    datas = pd.concat(datas, axis=1)
    return datas.reset_index()

class jq_bar_api():
    def __init__(self):
        self.all_factors = ['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor']
    def query(
        self,  
        universe=None,
        factor_list=None,
        start_date=None,
        end_date=None,
        trade_date=None,
    ):
        if trade_date is not None:
            trade_date = datetime.datetime.strptime(trade_date,'%Y-%m-%d')
            start_date = trade_date
            end_date = trade_date + datetime.timedelta(1)
            end_date = end_date.strftime('%Y-%m-%d')
        else:
            if start_date is None:
                start_date = '2010-01-01'

            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        df = jq.get_price(universe, start_date=start_date, end_date=end_date, frequency='daily', fields=factor_list, skip_paused=False, fq='post').rename(columns = {'time':'date'})
        return df



class ext_bardata_api2_jq():
    def __init__(self):
        self.all_factors = ['pe_ratio', 'turnover_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'capitalization', 'market_cap', 'circulating_cap', 'circulating_market_cap', 'pe_ratio_lyr']
    def query(
        self,  
        universe=None,
        factor_list=None,
        start_date=None,
        end_date=None,
        trade_date=None,
    ):
        if trade_date is not None:
            trade_date = datetime.datetime.strptime(trade_date,'%Y-%m-%d')
            start_date = trade_date
            end_date = trade_date + datetime.timedelta(1)
            end_date = end_date.strftime('%Y-%m-%d')
        else:
            if start_date is None:
                start_date = '2010-01-01'

            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # universe_set =tuple(universe)
        # sql = f"SELECT stock_valuation.id, stock_valuation.code, stock_valuation.pe_ratio, stock_valuation.turnover_ratio, stock_valuation.pb_ratio, stock_valuation.ps_ratio, stock_valuation.pcf_ratio, stock_valuation.capitalization, stock_valuation.market_cap, stock_valuation.circulating_cap, stock_valuation.circulating_market_cap, stock_valuation.day, stock_valuation.pe_ratio_lyr \nFROM stock_valuation \nWHERE stock_valuation.code in {universe_set} AND ((stock_valuation.day>='{start_date}') AND (stock_valuation.day<='{end_date}')) \n LIMIT 10000"
        # sql2 = "SELECT stock_valuation.id, stock_valuation.code, stock_valuation.pe_ratio, stock_valuation.turnover_ratio, stock_valuation.pb_ratio, stock_valuation.ps_ratio, stock_valuation.pcf_ratio, stock_valuation.capitalization, stock_valuation.market_cap, stock_valuation.circulating_cap, stock_valuation.circulating_market_cap, stock_valuation.day, stock_valuation.pe_ratio_lyr \nFROM stock_valuation \nWHERE stock_valuation.code IN ('000001.XSHE') AND stock_valuation.day = '2022-08-01' \n LIMIT 10000"
        # ins = JQDataClient.instance()
        # ins.get_fundamentals(sql=sql)
        
        days = jq.get_trade_days(start_date=start_date,end_date=end_date, count=None)
        all_df = []
        for day in days:
            
            df = jq.get_fundamentals(jq.query(
                jq.valuation
            ).filter(
                jq.valuation.code.in_(universe)
            ), date=day)

            if df.empty:
                print(f'{day} empty')
                continue
            del df['id']
            df.rename(columns={'day': 'date'}, inplace=True)

            columns = list(df.columns)
            columns.insert(0, columns.pop(columns.index('code')))
            columns.insert(0, columns.pop(columns.index('date')))
            df0 = df[columns]
            all_df.append(df0)
        all_df = pd.concat(all_df)  
        all_df = all_df.set_index(['date','code'])[self.all_factors]
        return all_df


class ext_bardata_api_jq():
    def __init__(self):
        self.all_factors = ['industry_name', 'capitalization', 'circulating_cap', 'eps_ttm', 'sz50', 'hs300', 'zz500', 'zz1000', 'kc50', 'szcz', 'cybz', 'szzs']
    def query(
        self,  
        universe=None,
        factor_list=None,
        start_date=None,
        end_date=None,
        trade_date=None,
    ):
        if trade_date is not None:
            trade_date = datetime.datetime.strptime(trade_date,'%Y-%m-%d')
            start_date = trade_date
            end_date = trade_date + datetime.timedelta(1)
            end_date = end_date.strftime('%Y-%m-%d')
        else:
            if start_date is None:
                start_date = '2010-01-01'

            if end_date is None:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
        all_data = get_summary_data(start_date=start_date,end_date=end_date,trade_date=trade_date)
        all_data = all_data.set_index(['date','code'])[self.all_factors]
        return all_data

jq_bar_api = jq_bar_api()
ext_bardata_api_jq = ext_bardata_api_jq()
ext_bardata_api2_jq = ext_bardata_api2_jq()


if __name__ == '__main__':
    all_sec = jq.get_all_securities().index.tolist()
    bar = jq_bar_api.query(universe=all_sec,start_date='2022-07-25',end_date='2022-07-25')
    ext_bardata = ext_bardata_api_jq.query(universe=all_sec,start_date='2022-07-25',end_date='2022-07-25')
    ext_bardata2 = ext_bardata_api2_jq.query(universe=all_sec,start_date='2022-07-25',end_date='2022-07-25')
    print('paused......')