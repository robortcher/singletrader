"""
constant
"""
from multiprocessing import cpu_count
import os 
# __all__ = ['CORE_NUM',
#         'root_dir','QLIB_BIN_DATA_PATH','AUTO_SAVE','MIN_SAVE_LENGTH','get_all_industry'
#     ]

CORE_NUM = min(16,cpu_count())


root_dir = r'~/.xlib'
QLIB_BIN_DATA_PATH = root_dir+'/qlib_data' 

AUTO_SAVE = True
MIN_SAVE_LENGTH = 800000  # 最小存储条数



