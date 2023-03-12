import sys
from pathlib import Path
current_filename = Path(__file__)
sys.path.append(current_filename.parent.parent.parent.__str__())

# sys.path.append(r'D:/projects/singletrader/')
from singletrader.datautils.qlibapi.dump.dump_bin import DumpAll, Update
from singletrader.datautils.dataapi.datasaver import UpdateWriter
from singletrader.constant import CONST

# last_trade_date = CONST.LAST_TRADE_DAY
current_trade_date = CONST.CURRENT_TRADE_DAY
if not CONST.IS_TRADE_DATE:
    print('今天不是交易日，无需运行数据更新')
    sys.exit()
if __name__ =='__main__':
    UpdateWriter(trade_date=current_trade_date)
    # DumpAll()
    Update(end_date=current_trade_date)