import pandas as pd
class BaseStrategy:
    def __init__(self,params={},**kwargs):
        self._params={}
        self._set_params(params=params,**kwargs)
        pass
    
    def start(self):
        print("strategy start...")
        pass
    
    def begin_bar(self):
        pass
    def run_bar(self,bar):
        pass
    def end_bar(self,bar):
        self.last_trade_date=bar.datetime
        pass
    
    def _set_params(self,params={},**kwargs):
        self._params.update(params)
        self._params.update(kwargs)
        for _key,_value in self._params.items():
            setattr(self,_key,_value)
    
    @property
    def name(self):
        pass
        
    
    def end(self):
        print("strategy end...")
        pass
    
    @property
    def Buy(self):
        return self.account.Buy
    
    @property
    def Sell(self):
        return self.account.Sell
    
    @property
    def order_to_shares(self):
        return self.account.order_to_shares
    
    @property
    def order_value(self):
        return self.account.order_value
    
    @property
    def order_to_ratio(self):
        return self.account.order_to_ratio
    
    @property
    def current_positions(self):
        return self.account.position.current_positions

    @property
    def cash(self):
        return self.account.position.cash

    @property
    def current_weights(self):
        return self.account.current_weights

    def __call__(self,**kwargs):
        return BaseStrategy(**kwargs)
    


class SingnalFactor(BaseStrategy):
    holding_period=1
    factor_name = 'pred_'
    parameters = ['target_buy_group']
    # method = 'long-short'
    method = 'only-long'
    insturment = None
    max_n = 50
    
    factor_value1_hist = pd.read_csv(r'D:/projects/singletrader/singletrader/model_data/alpha158/preds/2023.csv')
    factor_value2_hist = pd.read_csv(r'D:/projects/singletrader/singletrader/model_data/alpha158_rank/preds/2023.csv')

    
    def run_bar(self, bar):
        am=self.am
        current_dt_str = bar.datetime.strftime('%Y-%m-%d')

        facotr_value1 = self.factor_value1_hist[self.factor_value1_hist['datetime']==self.last_date]
        factor_value2 = self.factor_value1_hist[self.factor_value1_hist['datetime']==self.last_date]
        factor_value = factor_value[factor_value['is_st']==0]['score']
        paused = am.paused.iloc[-1]




        positions = list(self.current_positions.keys())
        for in_pos in positions:
            if in_pos not in target_cons:
                self.account.order_to_ratio(in_pos, 0, price = bar.close[in_pos]*(1-0.002))
        target_cash = 46000*1000#self.account.cash / target_cons.__len__()

        positions = list(self.current_positions.keys())
        for stk in target_cons: 
            if stk in positions:
                continue 
            ex_price = bar.open[stk] if not bar_['a2_p'][stk]>0 else float(bar_['a1_p'][stk])
            self.account.order_value(stk,target_cash, price = ex_price+0.0)
        self.last_date = current_dt_str

if __name__ == '__main__':
    sf = SingnalFactor()
    sf. run_bar(bar = None)
  






    
