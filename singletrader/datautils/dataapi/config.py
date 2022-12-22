class DataBaseConfig():
    # name = None
    add_preffix = None
    table_name = None
    need_lower = True
    host = '127.0.0.1'

class MysqlConfig(DataBaseConfig):
    sql_manager = 'mysql' #sql数据库类型
    pysql_package = 'pymysql' #sql python包
    port = '3306'
    username = 'root'
    password = 'xiao19940929'
    

class PostgresqlConfig(DataBaseConfig):
    sql_manager = 'postgresql' #sql数据库类型
    pysql_package = 'psycopg2' #sql python包
    port = '5432'


class ValuationConfig(MysqlConfig):
    
    db_name = 'finance'
    table_name = 'cn_jq_summary'
