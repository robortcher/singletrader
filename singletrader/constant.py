
import datetime
from datetime import timedelta
import pandas as pd
from functools import partial
import os
from multiprocessing import cpu_count
from .shared.utility import load_pkl,save_pkl

# 常量化存储
home_path ='D:\database'#os.environ['USERPROFILE']

# 环境变量化存储
# def init_path(path='D:\database'):
#     # path='D:\database'
#     command =r"setx HOME_PATH_SG %s /m"%path
#     command2 =r"setx HOME_PATH_SG %s "%path
#     os.system(command)
#     os.system(command2)
# home_path = os.os.environ['HOME_PATH_SG']

CORE_NUM = min(16,cpu_count()) # 模型使用核数
root_dir = home_path + '/' + '.singletrader' # 项目数据根目录
IND_PATH = root_dir+'/'+'ind_data' # 行业数据存储路径
TRADE_DATE_PATH  = root_dir + '/' + 'trade_date.pkl' # 交易日存取路径
QLIB_BIN_DATA_PATH = root_dir+'/'+'qlib_data'  # lib数据存储路径
os.environ["NUMEXPR_MAX_THREADS"] = "8"



current_date = datetime.datetime.now()
last_day = current_date - timedelta(1)


# def get_trade_days()
def get_trade_days(start_date=None, end_date=datetime.datetime.now().strftime('%Y-%m-%d'), count=None):
    if start_date is not None:
        start_datetime = datetime.datetime.strptime(start_date,'%Y-%m-%d').date()
    else:
        start_datetime = datetime.datetime.strptime('2005-01-01','%Y-%m-%d').date()
    
    if end_date is not None:
        end_datetime = datetime.datetime.strptime(end_date,'%Y-%m-%d').date()
    else:
        end_datetime = datetime.datetime.now().date()

    if os.path.exists(TRADE_DATE_PATH):
        all_dates = load_pkl(TRADE_DATE_PATH)
    
    else:
        import jqdatasdk as jq
        jq.auth(os.environ.get('JQ_USER') ,
            os.environ.get('JQ_PASSWD'))
        all_dates = jq.get_trade_days(start_date='2005-01-01',end_date='2050-12-31')
        save_pkl(all_dates,TRADE_DATE_PATH)
        return get_trade_days(start_date, end_date, count)
    
    all_dates = [date for date in all_dates if date >= start_datetime and date <= end_datetime]
    if count is not None:
        if end_date is None:
            return all_dates[:count]
        elif start_date is None:
            return all_dates[-count:]
    else:
        return all_dates



class CONST():
    """市场常用常量：交易日/上个交易日..."""
    LAST_DAY =  (datetime.datetime.now()-timedelta(1)).strftime('%Y-%m-%d')
    CURRENT_DAY = datetime.datetime.now().strftime('%Y-%m-%d')
    LAST_TRADE_DAY = get_trade_days(count=1,end_date = LAST_DAY)[-1].strftime('%Y-%m-%d')
    CURRENT_TRADE_DAY = get_trade_days(count=1,end_date = CURRENT_DAY)[-1].strftime('%Y-%m-%d')
    IS_TRADE_DATE = 1 if CURRENT_DAY == CURRENT_TRADE_DAY  else 0
    IS_TRADE_DATE_LAST_DAY = 1 if LAST_DAY  ==  LAST_TRADE_DAY else 0



def get_industry_cons(start_date=None, end_date=None, trade_date=current_date, name='sw_l1'):
    ############行业成分##########

    all_trade_dates = get_trade_days(count=1)
    path = IND_PATH + '/'+all_trade_dates[0].strftime('%Y-%m-%d') + '_'+ name +'.pkl'
    
    if os.path.exists(path):
            return load_pkl(path)
    
    def __get_daily_cons(ind, date, ind_names):
        ind_name = ind_names[ind]
        ind_cons = jq.get_industry_stocks(ind, date=date)
        ind_names = [ind_name] * len(ind_cons)
        dates_index = [date] * len(ind_cons)
        df = pd.DataFrame(
            {'date': dates_index, 'code': ind_cons, 'industry_name': ind_names})
        return df
    
    import jqdatasdk as jq
    jq.auth(os.environ.get('JQ_USER') ,
        os.environ.get('JQ_PASSWD'))
    res = pd.DataFrame()
    for date in all_trade_dates:

        universe = jq.get_all_securities(date=date).index
        all_sw_industry = jq.get_industries(name=name, date=date)
        all_sw_industry2 = all_sw_industry.reset_index()

        ind_names = all_sw_industry['name']

        func = partial(__get_daily_cons, date=date, ind_names=ind_names)
        ind_cons = all_sw_industry2.iloc[:, 0].apply(func)
        ind_cons = pd.concat(ind_cons.values).reset_index(drop=True)

        res = pd.concat([res, ind_cons]).set_index('code')['industry_name']

    save_pkl(res,path)

    return res



Ind_info = get_industry_cons()

if __name__ == '__main__':
    ad = get_trade_days()
    print()
