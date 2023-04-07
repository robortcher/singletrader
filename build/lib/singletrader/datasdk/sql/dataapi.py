import pandas as pd
import datetime
import logging
import time
from singletrader.datasdk.sql.config import *
from sqlalchemy import create_engine
from singletrader import __date_col__,__symbol_col__





class BaseSqlApi():
    """
    单表单列数据接口
    """
    def __init__(self, db_config=None, symbol_col=__symbol_col__, date_col=__date_col__):
        self.db_config = db_config
        self.symbol_col = symbol_col
        self.date_col = date_col
        self.table_name = db_config.table_name
        self.engine = create_engine(f'{self.db_config.sql_manager}+{self.db_config.pysql_package}://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.db_name}')
        # self.name = db_config.name
        self.add_preffix = db_config.add_preffix
        self.all_factors = self.get_all_factors()
        self.__all_factors = None
    
    # @property
    def get_all_factors(self):
        engine = self.engine
        if self.table_name is None:
            sql = f"SELECT table_name FROM information_schema.tables WHERE table_schema='public'"         
            res = pd.read_sql(sql = sql, con = engine)
            res = res['table_name'].tolist()
            if self.add_preffix is not None:
                preffix_n = len(self.add_preffix.strip())
                res  = [i[preffix_n:] for i in res if self.add_preffix in i]
        else:
            if self.db_config.sql_type=='mysql':
                sql = f"select column_name from information_schema.columns where table_schema='{self.db_config.db_name}' and table_name='{self.table_name}'"
            elif self.db_config.sql_type=='pgsql':
                sql = f"select column_name from information_schema.columns where table_schema='public' and table_name='{self.table_name}'"
            res = pd.read_sql(sql = sql, con = engine)
            res.columns = [_col.lower() for _col in res.columns]
            res = res[~res.isin([self.date_col,self.symbol_col,'update_time'])].dropna()['column_name'].tolist()
        self.__all_factors = res
        engine.dispose()
        return res
    
    def __call__(
        self, 
        date_range=None,
        universe = None,
        factor_list = None,
        start_date = None,
        end_date = None,
        trade_date = None,
        fileds = None,
    ):
        universe_mark = universe
        if date_range is not None:
            start_date = date_range[0]
            end_date = date_range[1]
        elif trade_date is not None:
            start_date = trade_date
            end_date = trade_date 
        else:
            if start_date is None:
                start_date = '2010-01-01'

            if end_date is None:
                end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if universe is not None:
            universe = sorted(universe)

        if factor_list is None:
            factor_list = self.get_all_factors()
        if self.db_config.sql_type=='mysql':
            date_filter = f"(`{self.date_col}` >= \'{start_date}\') and (`{self.date_col}` <= \'{end_date}\')"
        elif self.db_config.sql_type=='pgsql':
            date_filter = f"(\"{self.date_col}\" >= \'{start_date}\') and (\"{self.date_col}\" <= \'{end_date}\')"
    
        
        #股票筛选
        if universe is not None:
            universe = ["\'" + i.strip() + "\'" for i in universe]
            universe_str =  ','.join(universe)
            if self.db_config.sql_type=='pgsql':
                symbol_filter   =  f"(\"{self.symbol_col}\" in ({universe_str}))"
            
            elif self.db_config.sql_type=='mysql':
                symbol_filter   =  f"(`{self.symbol_col}` in ({universe_str}))"

        else:
            symbol_filter = 'True'
        

        #datas = []  
        datas = pd.DataFrame()
        factor_list_ = sorted(factor_list)
        # UNIQUE_KEY = hashlib.md5(f'{self.name}_{start_date}_{end_date}_{universe_mark}_{factor_list_}'.encode('utf-8')).hexdigest()

        
        engine = self.engine
        if self.table_name is None:
            for factor in factor_list:                
                if self.db_config.need_lower:
                    table_name = factor.lower()
                else:
                    table_name = factor
                if self.add_preffix is not None:
                    table_name = self.add_preffix + table_name
                fields_str = '*'
                sql = f"select {fields_str} from {table_name} where {date_filter} and {symbol_filter}"
                i = 0
                
                while i <3:
                    try:
                        # logging.info(f"正在下载{start_date}-{end_date}数据")
                        cur_df = pd.read_sql(sql = sql, con= engine)
                        if cur_df.__len__() == 0:
                            # logging.info(f"length of {factor} data is 0")
                            i+=1
                            # logging.info(f"第{i}次获取失败,重新获取。。。")
                            continue
                        cur_df[self.date_col] =  pd.to_datetime(cur_df[self.date_col])
                        cur_df = cur_df.rename(columns = {self.symbol_col:'code', self.date_col:'date'})
                        
                        cur_df = cur_df.set_index(['date', 'code'])
                        cur_df.columns = [i.lower() for i in cur_df.columns]
                        cur_df = cur_df[factor.lower()]
                        cur_df.name = factor
                    
                            
                        cur_df = cur_df[~cur_df.index.duplicated()]
                        datas[factor] = cur_df
    
                        
                    except Exception as e:
                        i+=1
                        # logging.info(f"第{i}次获取失败,重新获取。。。")
                        if i == 3:
                            logging.info(f"{factor} download failed")
        
        else:
            table_name = self.table_name
            cur_fields = [self.date_col, self.symbol_col]+ factor_list
            # fields_str = ','.join(["\"" + i + "\"" for i in cur_fields])
            fields_str = '*'
            sql = f"select {fields_str} from {table_name} where {date_filter} and {symbol_filter}"
            i = 0
            while i < 3:
                try:
                    cur_df = pd.read_sql(sql = sql, con= engine)

                    cur_df[self.date_col] =  pd.to_datetime(cur_df[self.date_col])
                    cur_df = cur_df.rename(columns = {self.symbol_col:'code', self.date_col:'date'})
                    cur_df = cur_df.set_index(['date', 'code'])
                    cur_df = cur_df[~cur_df.index.duplicated()]
                    if cur_df.__len__() == 0:
                        i += 1
                        # logging.info(f'第{i}次获取数据失败，尝试重新获取...')
                        time.sleep(2)
                        continue
                    datas = cur_df
                    # logging.info(f'{start_date}->{end_date}的{self.name}数据下载完毕')
                    break
                except Exception as e:
                    i += 1
                    if i == 3:
                        logging.info("第3次获取数据失败...")
        engine.dispose()
        
        if datas.__len__() == 0:
            datas = pd.DataFrame()
            return datas
        datas = datas.sort_index()
        return datas

get_valuation = BaseSqlApi(db_config=ValuationConfigPG)
get_valuation.all_factors = ['pe_ratio', 'turnover_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'capitalization', 'market_cap', 'circulating_cap', 'circulating_market_cap', 'pe_ratio_lyr']

get_price = BaseSqlApi(db_config=PricePostConfigPG)
get_price.all_factors = ['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor']


get_acution = BaseSqlApi(db_config=AuctionConfigPG)

get_index_cons = BaseSqlApi(db_config=IndexCons)

get_income = BaseSqlApi(db_config=IncomeConfig)
get_balance = BaseSqlApi(db_config=BalanceConfig)
get_cash_flow = BaseSqlApi(db_config=CashflowConfig)


if __name__ == '__main__':
    d0 = get_index_cons(start_date='2005-01-01')
    d1 = get_income()
    d2 = get_balance()
    d3 = get_cash_flow()
    d5 = get_valuation(trade_date='2023-03-15')
    print('end')