from .bar import DataH
from .account import Account
import logging

import pandas as pd
from .shared import ArrayManger

class Engine:
    def __init__(
        self,
        universe=None,
        start_date='2022-01-01',
        end_date='2022-01-31',
        **kwargs
        
    ):
        self.unverse=universe
        self.start_date=start_date
        self.end_date=end_date
        self._set_params(**kwargs)
        self._accounts = []

    def load_data(self,src,**kwargs):
        self._dataH=DataH(src)
        self._dataH.download(
                                symbol_list=self.unverse,
                                start_date=self.start_date,
                                end_date=self.end_date,
                                **kwargs
                            )
        
        self._data_kwargs = kwargs
    def _set_params(self,**kwargs):
        for _k,_v in kwargs.items():
            setattr(self,_k,_v)
    
    def run_backtest(self,strategy,account,benchmark=None,expanding=False,including_current=False, **kwargs):
        am_size = kwargs.pop('am_size',1)
        _strategy = strategy(**kwargs)
        _strategy.account = account
        _strategy.am = ArrayManger(size=am_size,universe=self._dataH.all_symbols,cols=self._dataH.all_cols,expanding=expanding)
        _strategy.start()
        _bar_generator = self._dataH.bar_generator
        while True:
            try:
                bar = next(_bar_generator)
                _strategy.begin_bar()
                if including_current:
                    _strategy.am.update_bar(bar)
                if _strategy.am.inited:
                    _strategy.account.init_bar(bar)
                    _strategy.account.update_begin_bar(bar)
                    _strategy.run_bar(bar)
                    _strategy.account.update_bar(bar)
                    _strategy.end_bar(bar)
                    _strategy.account.update_end_bar(bar)
                if not including_current:
                    _strategy.am.update_bar(bar)
                
            except StopIteration:
                _strategy.account.hist_records = pd.DataFrame(_strategy.account.hist_orders).set_index('datetime')
                _strategy.account.hist_total_asset = pd.DataFrame(_strategy.account.hist_total_asset,columns=['datetime','total_asset']).set_index('datetime')['total_asset']
                _strategy.account.daily_amount = _strategy.account.hist_records['amount'].groupby(level=0).sum()
                _strategy.account.net_value = _strategy.account.hist_total_asset / _strategy.account.hist_total_asset.iloc[0]
                _strategy.account.net_value_bm = self.get_benchmark_nv(benchmark=benchmark)
                _strategy.account.net_value_bm =_strategy.account.net_value_bm.reindex(_strategy.account.net_value.index) / _strategy.account.net_value_bm.reindex(_strategy.account.net_value.index).iloc[0]
                self._accounts.append(_strategy.account)
                return _strategy.account
    
    
    def get_benchmark_nv(self,benchmark=None,price_type='close',**kwargs):
        if benchmark is None:
            benchmark = 'avg'
            _d = self._dataH._data.set_index([self._dataH.date_col,self._dataH.symbol_col])[price_type].unstack()
            
        else:
            kwargs.update(self._data_kwargs)
            if type(benchmark) is str:
                symbol_list = [benchmark]
            _d = self._dataH.download(
                                    symbol_list=symbol_list,
                                    start_date=self.start_date,
                                    end_date=self.end_date,
                                    just_data=True,
                                    **kwargs
                                )
            _d = _d.set_index([self._dataH.date_col,self._dataH.symbol_col])[price_type].unstack() 
        
        benchmark_return = _d.pct_change().mean(axis=1)
        benchmark_return.name = benchmark
        return (1+benchmark_return.fillna(0)).cumprod()
    
    def get_all_info(self, key='met_value'):
        accts_num = self._accounts.__len__()
        assert accts_num > 0, '并未进行任何回测，请先使用Engine.run_backtest函数运行回测'
        df = pd.concat([getattr(self._accounts[_i], key) for _i in range(accts_num)],axis=1)
        df.columns = list(range(accts_num))
        return df
    
    
    def get_all_nv(self):
        accts_num = self._accounts.__len__()
        assert accts_num > 0, '并未进行任何回测，请先使用Engine.run_backtest函数运行回测'
        df = pd.concat([self._accounts[_i].net_value for _i in range(accts_num)],axis=1)
        df.columns = list(range(accts_num))
        return df
    
    def get_hist_weights(self):
        accts_num = self._accounts.__len__()
        assert accts_num > 0, '并未进行任何回测，请先使用Engine.run_backtest函数运行回测'
        df =  pd.concat({_i:pd.DataFrame(self._accounts[_i].hist_weights) for _i in range(accts_num)},axis=1).T
        return df 

    

if __name__ == '__main__':
    from strategy import BaseStrategy,testStrategy,AutoAlloStrategy
    # engine = Engine(universe=['IVV','SCHD','AGG','SCHP'],strategy=Strategy)
    engine = Engine(universe=['IVV','SCHD','AGG','SCHP'],start_date='2010-01-01',end_date='2022-12-31')
    engine.load_data(src='yahoo',interval="1mo")
    w_mkt_dict = {
        'IVV':0.25,#['SPY','SFY','IVV'],0.03%
        'SCHD':0.25,#['SCHD','SDY','DVY'],0.06%
        'AGG':0.25,#['AGG','BND'], 0.03%
        'SCHP':0.25,#['TIP','VTIP','SCHP'] 0.04%
    }
    w_mkt = pd.Series(w_mkt_dict)


    
    accts = []
    for i in range(6):
        account = Account(
            capital_base = 1e8,
            commission_buy = 0.000,
            commission_sell = 0.000,
            margin_ratio = 1,
            min_commission = 0.,
            trade_free = False,
            price_type = 'close',
            slippage = 0.,
            slippage_ratio = 0.,
            tax_buy = 0.,
            tax_sell = 0.00,
            region='us',
        )
        acct = engine.run_backtest(strategy=AutoAlloStrategy, 
                                account=account,
                                am_size=12*6,
                                benchmark='SPY', 
                                w_mkt=w_mkt,
                                total_risk_level=6,
                                user_risk_level=i
                                )
        accts.append(acct)
    print('===')
            