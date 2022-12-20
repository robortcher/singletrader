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
    


class testStrategy(BaseStrategy):
    def run_bar(self,bar):
        if 'IVV' not in self.current_positions:
            self.order_to_ratio('IVV',1)
        pass




    
