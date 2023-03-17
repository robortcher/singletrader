"""下载历史数据和更新每日数据的脚本"""

import sys
from pathlib import Path
import sys
current_filename = Path(__file__)
sys.path.append(current_filename.parent.parent.parent.__str__())

# sys.path.append(r'D:/projects/singletrader/')
from singletrader.datasdk.qlib.dump_bin import DumpAll,Update
from singletrader.datasdk.sql.datasaver import UpdateWriter
from singletrader.constant import CONST



mode = 'update'

# last_trade_date = CONST.LAST_TRADE_DAY
current_trade_date = CONST.CURRENT_TRADE_DAY
if not CONST.IS_TRADE_DATE:
    print('今天不是交易日，无需运行数据更新')
    sys.exit()
if __name__ =='__main__':
    
    
    if mode == 'all':
        DumpAll()
    if mode == 'update':
        UpdateWriter(trade_date=current_trade_date)
        Update(end_date=current_trade_date)