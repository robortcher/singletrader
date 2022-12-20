from collections import defaultdict
from dataclasses import dataclass
import datetime
from .bar import Bar
from .order import Order
import numpy as np

@dataclass
class PositionStock:
    symbol:str
    datetime:datetime=None
    shares:int=0
    shares_yesterday=0
    price_type = 'close'
    last_price:float=0
    def __post_init__(self):
        self.inited:bool=False
        self.net_cashflow=0
        # self.last_price:float=0

    def update_bar(self,bar:Bar):
        """update after new bar"""
        self.last_price=getattr(bar,self.price_type)[self.symbol]
        self.datetime=bar.datetime
        self.inited=True 

    def update_order(self,order:Order):
        """update after new order"""
        if order.status:
            self.shares += (order.shares * np.sign(order.type.value))
            self.net_cashflow += order.net_cash_flow

            
    @property
    def market_value(self):
        return self.last_price * self.shares
    
    @property
    def avg_cost(self):
        return -self.net_cashflow / self.shares
    
    @property
    def value(self):
        pass


class Position():
    def __init__(self):
        self.current_positions = {}
        self.hist_positions = {}
        pass
    
    def check_in(self,symbol):
        if self.current_positions[symbol].shares == 0:
            del self.current_positions[symbol]
            return False
        else:
            return True
    
    def _init(self,symbol,datetime):
        price = self.last_begin_price[symbol]
        self.current_positions[symbol] = PositionStock(symbol=symbol,datetime=datetime,last_price=price)
    
    def update_begin_bar(self,bar,price_type='open'):
        self.last_begin_price=getattr(bar,price_type)

    
    def update_bar(self,bar):
        for symbol in self.current_positions:
            if self.check_in(symbol):
                self.current_positions[symbol].update_bar(bar)
                self.hist_positions[bar.datetime] = self.current_positions
    
    
    def update_order(self,order:Order):
        if order.status:
            if order.symbol not in self.current_positions:
                self._init(order.symbol,order.datetime)
            self.current_positions[order.symbol].update_order(order)
            self.check_in(order.symbol)   


    @property
    def total_mv(self):
        return sum([self.current_positions[_p].market_value for _p in self.current_positions])
    
    def __getitem__(self,key):
        return self.current_positions[key]

    def _data(self):
        pass