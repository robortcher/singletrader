"""
constant
"""
from multiprocessing import cpu_count
import os 
from singletrader.shared.logging import logger




def check_and_mkdir(path):
    # logger.info("开始初始化路径...")
    if not os.path.exists(path):
        os.mkdir(path)
        # logger.info(f"开始创建路径{path}")

def init_and_check_jq():
    
    # logger.info("开始初始化JQ账号...")
    jq_user = os.environ['JQ_USER']
    jq_passwd = os.environ['JQ_PASSWD']
    
    if jq_user is None or jq_passwd is None:
        # logger.info(r"未初始化聚款账号，部分功能无法正常")
        # logger.info(r"请使用JQ_USER和JQ_PASSWD作为变量名，设置相应环境变量")
        return False
    else:
        return True


    

home_path = os.environ['HOME']
CORE_NUM = min(16,cpu_count())
root_dir = home_path + '/' + '.singletrader'
QLIB_BIN_DATA_PATH = root_dir+'/'+'qlib_data' 

def init():
    if int(os.environ.get('singletrader.init',0)):
        pass 
    else:
        logger.info("开始初始化路径...")
        check_and_mkdir(root_dir)
        check_and_mkdir(QLIB_BIN_DATA_PATH)
        
        logger.info("开始初始化JQ账号...")
        if not init_and_check_jq():
            logger.info(r"未初始化聚款账号，部分功能无法正常")
            logger.info(r"请使用JQ_USER和JQ_PASSWD作为变量名，设置相应环境变量")
        else:
            logger.info(r"聚宽账号初始化成功...")
        os.environ['singletrader.init'] = '1'

        
init()


