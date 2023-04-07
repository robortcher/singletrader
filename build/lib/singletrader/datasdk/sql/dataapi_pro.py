import psycopg2
import threading
from config import PricePostConfigPG
# 创建连接池
connection_pool = psycopg2.pool.SimpleConnectionPool(
    1,  # 最小连接数
    10,  # 最大连接数
    user=PricePostConfigPG.username,
    password=PricePostConfigPG.password,
    host=PricePostConfigPG.host,
    port=PricePostConfigPG.port,
    database=PricePostConfigPG.db_name
)

def read_data(thread_name):
    connection = connection_pool.getconn()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM your_table")
    data = cursor.fetchall()
    print(f"{thread_name}: {data}")
    cursor.close()
    connection_pool.putconn(connection)

# 创建并启动线程
threads = []
for i in range(5):
    thread = threading.Thread(target=read_data, args=(f"Thread {i}",))
    threads.append(thread)
    thread.start()

# 等待所有线程执行完毕
for thread in threads:
    thread.join()

# 关闭连接池
connection_pool.closeall()
