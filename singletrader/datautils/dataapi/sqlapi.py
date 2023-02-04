import pandas as pd
import datetime
import logging
import time
from singletrader.datautils.dataapi.config import ValuationConfig,SummaryConfig,ValuationConfigPG,SummaryConfigPG
from sqlalchemy import create_engine
class data_api_mode():
    """
    单表单列数据接口
    """
    def __init__(self, db_config = None, symbol_col = 'code', date_col = 'date'):
        self.db_config = db_config
        self.symbol_col = symbol_col
        self.date_col = date_col
        self.table_name = db_config.table_name
        self.engine = create_engine(f'{self.db_config.sql_manager}+{self.db_config.pysql_package}://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.db_name}')
        # self.name = db_config.name
        self.add_preffix = db_config.add_preffix
        self.all_factors = self.get_all_factors()
        self.__all_factors = None
        
    def __call__(self):
        return self

    # @property
    # def engine(self):
    #     from sqlalchemy import create_engine
    #     return create_engine(f'{self.db_config.sql_manager}+{self.db_config.pysql_package}://{self.db_config.username}:{self.db_config.password}@{self.db_config.host}:{self.db_config.port}/{self.db_config.db_name}')

    
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
    
    def query(
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

ext_bardata_api = data_api_mode(db_config=SummaryConfig)
ext_bardata_api.all_factors = ['industry_name', 'capitalization', 'circulating_cap', 'eps_ttm', 'sz50', 'hs300', 'zz500', 'zz1000', 'kc50', 'szcz', 'cybz', 'szzs']
ext_bardata_api2 = data_api_mode(db_config=ValuationConfig)
ext_bardata_api2.all_factors = ['pe_ratio', 'turnover_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'capitalization', 'market_cap', 'circulating_cap', 'circulating_market_cap', 'pe_ratio_lyr']



ext_bardata_api_pg = data_api_mode(db_config=SummaryConfigPG)
ext_bardata_api_pg.all_factors = ['industry_name', 'capitalization', 'circulating_cap', 'eps_ttm', 'sz50', 'hs300', 'zz500', 'zz1000', 'kc50', 'szcz', 'cybz', 'szzs']
ext_bardata_api2_pg = data_api_mode(db_config=ValuationConfigPG)
ext_bardata_api2_pg.all_factors = ['pe_ratio', 'turnover_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio', 'capitalization', 'market_cap', 'circulating_cap', 'circulating_market_cap', 'pe_ratio_lyr']


def get_security_info(**kwargs):
    instruments = ext_bardata_api.query(trade_date='2022-12-15')
    instruments = instruments.reset_index()
    if 'start_date' not in instruments.columns:
        instruments['start_date'] = pd.to_datetime('2005-01-01').date()
    if "end_date" not in instruments.columns:
        instruments['end_date'] = pd.to_datetime('2030-01-01').date()
    return instruments

def get_trade_days(start_date='2010-01-01',end_date="2022-12-31"):
    dates = pd.date_range(start_date,end_date)
    return list(map(lambda x:x.date(),dates))



if __name__ == '__main__':
    a = get_security_info()
    d = get_trade_days()
    from config import ValuationConfig,SummaryConfig,ValuationConfigPG
    value_api = data_api_mode(db_config=ValuationConfigPG)
    value_api.all_factors = ['industry_name', 'capitalization', 'circulating_cap', 'eps_ttm', 'sz50', 'hs300', 'zz500', 'zz1000', 'kc50', 'szcz', 'cybz', 'szzs']
    value_api.get_all_factors()
    d1 = value_api.query(trade_date='2022-12-15')
    
    summary_api = data_api_mode(db_config=SummaryConfig)
    d2 = summary_api.query(trade_date='2022-12-15')
    value_api= data_api_mode(db_config=ValuationConfig)
    d1 = value_api.query(trade_date='2022-12-15')
    import tushare as ts 
    print(d1)