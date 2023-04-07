import os
from singletrader import QLIB_BIN_DATA_PATH
from singletrader.datasdk.qlib.base import MultiFactor


feature_path = QLIB_BIN_DATA_PATH + '/features' + '/000001.XSHE'
all_features = [file[:-8] for file in os.listdir(feature_path)]
all_fields = ['$'+feature for feature in all_features]





if __name__ == '__main__':
    
    all_data = MultiFactor(field=all_fields,name=all_features,start_date='2015-01-01',end_date='2022-12-31')._data
    print('==')
    pass