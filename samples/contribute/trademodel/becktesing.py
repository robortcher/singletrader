from singletrader.shared.utility import load_pkls
from singletrader.backtesting.engine import Engine
from singletrader.backtesting.strategy import BaseStrategy

from singletrader.backtesting.account import Account
from singletrader.performance.common import performance_indicator 

singnals = load_pkls(r'D:\projects\singletrader_pro\samples\contribute\trademodel\predict500')


class SingnalFactor(BaseStrategy):
    holding_period=1
    factor_name = 'pred_'
    parameters = ['target_buy_group']
    # method = 'long-short'
    method = 'only-long'
    insturment = None
    max_n = 10
    count_n = 0
    period=5
    def run_bar(self, bar):
        if self.count_n % self.period ==0:
            am=self.am
            current_dt_str = bar.datetime.strftime('%Y-%m-%d')

            singnal = singnals[singnals.index.get_level_values('datetime')==self.last_trade_date.strftime('%Y-%m-%d')].dropna().droplevel('datetime')
            singnal = singnal[bar.paused.dropna()==0]
            target_cons = singnal.sort_values(ascending=False).index.tolist()
            target_cons = [i for i in target_cons if (not i.startswith('688')) and (not i.startswith('3'))]
            target_cons = target_cons[:self.max_n]

            positions = list(self.current_positions.keys())
            for in_pos in positions:
                if in_pos not in target_cons:
                    self.account.order_to_ratio(in_pos, 0, price = bar.avg[in_pos])
                else:
                    target_cons.remove(in_pos)

            
            target_cash = self.account.cash / target_cons.__len__()

            positions = list(self.current_positions.keys())
            for stk in target_cons: 
                if stk in positions:
                    continue 
                self.account.order_value(stk,target_cash, price = bar.avg[stk])
        
        # self.last_date = current_dt_str
        self.count_n+=1
accts = []
engine = Engine(initial_cash=1000000,start_date='2018-01-02',end_date='2023-08-23',slippage_ratio=0.00,Strategy=SingnalFactor,open_commission=0.00011, close_commission=0.00011, close_tax=0.001, benchmark=None,fp = False,price_type='close_price',trade_realse=False)
engine.load_data()

for s in ['2018-01-02','2018-01-03','2018-01-04','2018-01-05','2018-01-08']:

    # start_date = s#min(all_data['datetime'])
    # end_date =  '2023-02-28'#datetime.datetime.now().strftime('%Y-%m-%d')
    # engine = Engine(initial_cash=1000000000000,start_date=start_date,end_date=end_date,slippage_ratio=0.002,Strategy= SingnalFactor,open_commission=0.0001, close_commission=0.0001, close_tax=0.001, benchmark=benchmark,fp = False,price_type='close_price',trade_realse=False)
    account = Account(
        capital_base = 1000000,
        commission_buy = 0.00015,
        commission_sell = 0.00165,
        margin_ratio = 1,
        min_commission = 0.,
        trade_free = False,
        price_type = 'close',
        slippage = 0.,
        slippage_ratio = 0.00,
        tax_buy = 0.,
        tax_sell = 0.00,
        region='cn',
    )


    
    engine.run_backtest(account=account,start_date=s,strategy=SingnalFactor,am_size=1)
    accts.append(engine._accounts[0])
print('===')
