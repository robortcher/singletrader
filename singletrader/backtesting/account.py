"""
account
"""
from .bar import Bar
from .position import Position
from .order import Order,OrderType
import pandas as pd
DEFAULT_CAPITAL_BASE = 1e6

class Account():
    """账户信息"""
    def __init__(
        self,
        capital_base = DEFAULT_CAPITAL_BASE,#初始金额
        commission_buy = 0.,
        commission_sell = 0.,
        margin_ratio = 1,
        min_commission = 0.,
        trade_free = False,
        price_type = 'close',
        slippage = 0.,
        slippage_ratio = 0.,
        tax_buy = 0,
        tax_sell = 0,
        **kwargs
    ):
        self.capital_base = capital_base 
        self.cash = capital_base
        self.trading_amount = 0 
        self.price_type = price_type
        self.position = Position()
        self.commission_buy = commission_buy
        self.commission_sell = commission_sell
        self.min_commission = min_commission
        self.slippage = slippage
        self.slippage_ratio = slippage_ratio
        self.tax_buy = tax_buy
        self.tax_sell = tax_sell
        self.inited = False
        self.hist_orders = []
        self.hist_total_asset = []
        self.hist_weights = {}
        self.last_bar = None
        self._set_params(**kwargs)
    
    def _set_params(self,**kwargs):
        for _key,_value in kwargs.items():
            setattr(self,_key,_value)    
    
    def init_bar(self,bar):
        if not self.inited:
            self.datetime = bar.datetime
            self.current_price = getattr(bar, self.price_type)  
            # self.position.init_bar(bar)
            self.inited = True  
    
    def update_begin_bar(self,bar):
        self.datetime = bar.datetime
        self.position.update_begin_bar(bar)
    
    def update_bar(self,bar):
        # self.datetime = bar.datetime
        self.current_price = getattr(bar, self.price_type)
        self.position.update_bar(bar)
        self.hist_total_asset.append((bar.datetime,self.total_asset))
    
    def update_end_bar(self,bar):
        self.hist_weights[bar.datetime] = self.current_weights
        self.last_bar = bar
    
    
    
    def update_order(self,order:Order):
        if order.status:
            self.position.update_order(order)
            self.cash += order.net_cash_flow
            self.hist_orders.append(order.value)
            self.trading_amount += abs(order.amount)

    def share_adjust(self,symbol,shares):
        """不同市场报单数量调整"""     
        region = getattr(self,'region','cn')
        if region == 'cn':
            if symbol.startswith('688'):
                if shares <200:
                    shares = 0
                else:
                    shares = shares // 1
            else:
                shares = shares // 100 * 100
        elif region == 'us':
            shares = shares // 1
        return shares
        
            
    def create_order(self, symbol,shares,price,order_type,commission=0):
        """"""
        return Order(symbol=symbol,datetime=self.datetime,type=OrderType(order_type),shares=shares,price=price,commission=commission, last_bar=self.last_bar )
    

    
    def Buy(self,symbol,shares,price=None):
        
        shares = self.share_adjust(symbol=symbol,shares=shares)
        if price is None:
            price = self.current_price[symbol]
        
        ex_price = (price + self.slippage) * (1 + self.slippage_ratio) #执行价
        commission = ex_price * (self.commission_buy + self.tax_buy) #总成本
        commission = max(commission,self.min_commission)

        need_cash = shares * ex_price + commission
        
        if self.cash < need_cash:
            shares = self.cash / (ex_price*(1+self.commission_buy + self.tax_buy))
            shares = self.share_adjust(symbol=symbol,shares=shares)
    
        if shares == 0:
            return 
        order = self.create_order(
            symbol,shares,price,1,commission
        )
        self.update_order(order)
        
    def Sell(self,symbol,shares,price=None):
        if price is None:
            price = self.current_price[symbol]
        
        if shares >= self.position[symbol].shares:
            shares = self.position[symbol].shares
        
        else:
            shares = self.share_adjust(symbol=symbol,shares=shares)
            if shares >= self.position[symbol].shares:
                shares = self.position[symbol].shares
        
        if shares == 0:
            return
        
        ex_price = (price - self.slippage) * (1 - self.slippage_ratio)
        commission = ex_price * (self.commission_sell + self.tax_sell)
        commission = max(commission,self.min_commission)
        order = self.create_order(
            symbol,shares,price,-1,commission
        )
        self.update_order(order)

    def order_to_shares(self, symbol, shares, price = None,):
        """买到多少手"""
        if symbol in self.position.current_positions:
            current_shares = self.position[symbol].shares 
        else:
            current_shares = 0
        delta_shares = shares - current_shares
          
        if shares > 0:
            if delta_shares > 0 and current_shares >= 0:
                self.Buy(symbol=symbol, shares=delta_shares, price=price)

            elif delta_shares < 0 and current_shares > 0:
                self.Sell(symbol=symbol, shares=abs(delta_shares), price=price)        
        
        elif shares == 0:
            if current_shares>0:
                self.Sell(symbol=symbol, shares=abs(delta_shares), price=price,)


    def order_value(self, symbol, value, price=None,):
        if price is None:
            price = self.current_price[symbol]
        if value > 0:
            shares = value / ((price+self.slippage) * (1+self.slippage_ratio)) // 1
            self.Buy(symbol=symbol, shares=shares, price=price,)
        
        elif value ==0:
            return
        
        elif value <0:
            shares = abs(value) / ((price -self.slippage) * (1 - self.slippage_ratio)) // 1
            self.Sell(symbol=symbol, shares=shares, price=price)
    
    
    #目标持仓比例下单
    def order_to_ratio(self, symbol, target_ratio, price=None,):
        """"""
        if price is None:
            price = self.current_price[symbol]
        
        target_shares = (self.total_asset * target_ratio) // price
        self.order_to_shares(symbol=symbol, shares=target_shares, price=price)
    
    @property
    def total_asset(self):
        return self.position.total_mv + self.cash
    
    @property
    def current_weights(self):
        mv = pd.Series({_symbol:self.position.current_positions[_symbol].market_value
                   for _symbol in self.position.current_positions})
        return mv / self.total_asset        

    @property
    def turnover_ratio(self):
        return self.hist_records['amount'].groupby(level='datetime').sum() / self.hist_total_asset

if __name__ == '__main__':
    test_acct = Account(capital_base=1e6)
    
    
    pass 
