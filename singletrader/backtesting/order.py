from dataclasses import dataclass
from enum import Enum
import datetime
import pandas as pd
import numpy as np
from typing import Any
class OrderType(Enum):
    Buy=1
    Sell= -1
    Short=-2
    Cover_Short=2

@dataclass
class Order:
    symbol:str
    
    datetime:datetime
    type:OrderType=OrderType(1)
    shares:float=0
    price:float=0
    info_type:str=0
    commission:float=0
    last_bar:Any = None
    
    def __post_init__(self):
        self.amount = abs(self.shares * self.price)
        self.status = False
        self.order_check()
    
    
    def get_value(self):
        pd.Series(
            index=['datetime','symbol','type','shares','price','info_type']
        )

    @property
    def net_cash_flow(self):
        if not self.status:
            return 0
        else:
            return -self.commission - self.shares * np.sign(self.type.value) * self.price
    
    
    def order_check(self,limit=0.095,price_col='close',volume_col='volume'):
        if self.shares==0:
            
            return
        if self.last_bar is not None:
            if limit is not None:
                price_yesterday = getattr(self.last_bar, price_col)[self.symbol]
                high,low = price_yesterday*(1+limit),price_yesterday*(1-limit)
                if self.price>high:
                    if self.type.value >0:
                        print(f'{self.datetime}--{self.symbol}股票涨停，无法买入')
                
                elif self.price<low:
                    if self.type.value < 0:
                        print(f'{self.datetime}--{self.symbol}股票跌停，无法卖出')
                    return
        
                # volume = getattr(self.last_bar,volume_col,1e10)
                # if volume==0:
                #     return
                else:
                    self.status=True
            else:
                self.status=True
        else:
            self.status=True
    
    @property 
    def value(self):
        return {'datetime':self.datetime,'symbol':self.symbol,'shares':self.shares,'type':self.type.name,'ex_price':self.price,'commission':self.commission,'amount':self.amount,'net_cash_flow':self.net_cash_flow}
        
    

        

            
