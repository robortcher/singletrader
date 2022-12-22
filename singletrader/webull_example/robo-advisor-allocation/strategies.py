from singletrader.backtesting.strategy import BaseStrategy
from utils.optimizer import get_efficient_frontier,get_bl_return_cov


class BlackLittermanAllocation(BaseStrategy):
    """
    parameters
    -------------------
    constraint: 必要参数，单一资产权重限制
    linspace: 和sector_constrain,二选一,基于风险划分风险等级的方法
    sector_constraint: 和sector_constrain,二选一, 并且has_sector_constraint设置为True,大类限制
    has_sector_constraint: 是否有大类约束, 与sector_constraint同时输入
    """
    def run_bar(self,bar):
        if len(self.am.close.dropna())<60: #当少于50个历史数据时，不开始组合配置
            return 
        
        # returns = self.am.close.pct_change()
        market_caps = self.w_mkt
        # returns = self.am.close.pct_change().reindex(market_caps.index,axis=1)
        # returns.iloc[0] =  (self.am.close/self.am.open - 1).iloc[0]
        returns = (self.am.close/self.am.open - 1).reindex(market_caps.index,axis=1)
        f, V = get_bl_return_cov(prices=returns,market_caps=market_caps,frequency=12)
            
  
        try:    
            if self.has_sector_constraint:
                _p, _w = get_efficient_frontier(constraint=self.constraint,expected_returns=f,cov_matrix=V,sector_constraint=self.sector_constraint,method='max_sharpe')
            else:
                _p, _w = get_efficient_frontier(linspace=[self.target_vol],constraint=self.constraint,expected_returns=f,cov_matrix=V)

        except Exception as e:
            print(bar.datetime,e)
            return 
        
        target_w = _w.iloc[0]
        target_w[target_w<=1e-3]=0
        target_w = target_w/target_w.sum()
        target_w = target_w[target_w>0]
        
        spread = (target_w - self.current_weights.reindex(target_w.index).fillna(0)).sort_values()
        positions = list(self.current_positions.keys())
        
        for _p in positions:
            if _p not in spread.index:
                self.order_to_ratio(_p,0)
        
        for _p in spread.index:
            self.order_to_ratio(_p,target_w[_p])

        pass
    
    def end_bar(self, bar):
        return super().end_bar(bar)


