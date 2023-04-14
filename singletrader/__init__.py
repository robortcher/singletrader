
import os
from .shared.logging import logger,logger_init
from .constant import *


__version__ = "0.2.0"
__author__ = "Simon Xiao"
__email__ = "robortcher@outlook.com"
__description__ = "for a integrated factor testing and backtesing framework"
__url__ = "https://github.com/robortcher/singletrader"

__date_col__ = 'date'
__symbol_col__ = 'code'


def check_and_mkdir(path):
    # logger.info("开始初始化路径...")
    if not os.path.exists(path):
        os.mkdir(path)

def init():
    if int(os.environ.get('singletrader.init',0)):
        pass 
    else:
        logger_init.info("开始初始化路径...")
        check_and_mkdir(root_dir)
        check_and_mkdir(QLIB_BIN_DATA_PATH)
        check_and_mkdir(IND_PATH)
        
    try:
        import qlib
        qlib.init(provider_uri=QLIB_BIN_DATA_PATH)
    except ImportError:
        logger_init.error("please install pyqlib first")

    try:
        import jqdatasdk as jq
        jq.auth(os.environ['JQ_USER'],os.environ['JQ_PASSWD'])
    except:
        logger_init.error('please install jqdatasdk and auth it first')
    
    logger_init.info(f"路径初始化完毕。初始化路径为{QLIB_BIN_DATA_PATH}")
    os.environ['singletrader.init'] = '1'

        
init()