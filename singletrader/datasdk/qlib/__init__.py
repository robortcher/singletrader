"""
数据管理工具包
"""
from multiprocessing import cpu_count
import os 
from singletrader.shared.logging import logger
import qlib



def check_and_mkdir(path):
    # logger.info("开始初始化路径...")
    if not os.path.exists(path):
        os.mkdir(path)
        # logger.info(f"开始创建路径{path}")

# def init_and_check_jq():
#     import jqdatasdk as jq
#     # logger.info("开始初始化JQ账号...")
#     jq_user = os.environ.get('JQ_USER')
#     jq_passwd = os.environ.get('JQ_PASSWD')
#     jq.auth(jq_user,jq_passwd)
    
#     if jq_user is None or jq_passwd is None:
#         # logger.info(r"未初始化聚款账号，部分功能无法正常")
#         # logger.info(r"请使用JQ_USER和JQ_PASSWD作为变量名，设置相应环境变量")
#         return False
#     else:
#         return True


    

home_path = 'D:\database'#os.environ['USERPROFILE']
CORE_NUM = min(16,cpu_count())
root_dir = home_path + '/' + '.singletrader_pro'
IND_PATH = root_dir+'/'+'ind_data' 
QLIB_BIN_DATA_PATH = root_dir+'/'+'qlib_data' 
os.environ["NUMEXPR_MAX_THREADS"] = "8"


