import sys 
sys.path.append(r'D:/projects/singletrader')
from singletrader.datautils.qlibapi.constructor import MultiFactor
# from singletrader import init
# init()
fields = ['$market_cap','1/$pb_ratio', '0.79*1/$pe_ratio+0.21*1/$pcf_ratio']
names = ['size','bp','ep']



# prices = [100,3,2,5,10,12,31,44,11,22,5]

# def get_max_profit_one(prices):
#     min_price = prices[0]
#     max_pnl = 0 
#     for _price in prices:
#         min_price = min(min_price,_price)
#         max_pnl = max(max_pnl,_price-min_price)
#     return max_pnl

# def get_max_profit_two(prices):
#     steps = len(prices)
#     max_pnl = 0
#     for i in range(steps-1):
#         part1 = prices[:i+1]
#         part2 = prices[i+1:]
#         max_pnl = max(get_max_profit_one(part1) + get_max_profit_one(part2),max_pnl)
#     return max_pnl
    
    
    


if __name__ == '__main__':
    import jqdatasdk as jq
    all_secs = jq.get_all_securities().index.tolist()
    all_sec = all_secs
    mf = MultiFactor(field=fields,name=names,start_date='2005-01-01',end_date='2022-12-31',instruments=all_sec)
    d = mf._data
    print('=====')