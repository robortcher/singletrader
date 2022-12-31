"""
constant
"""
from multiprocessing import cpu_count
import os 
import logging


def check_and_mkdir(path):
    """检查目标路径是否存在，并创建新路径"""
    if not os.path.exists(path):
        os.mkdir(path)
        logging.info('make dir successfully....')


home_path = os.environ['HOME']
CORE_NUM = min(16,cpu_count())
root_dir = home_path + '/' + '.singletrader'
QLIB_BIN_DATA_PATH = root_dir+'/'+'qlib_data' 

check_and_mkdir(root_dir)
check_and_mkdir(QLIB_BIN_DATA_PATH)






