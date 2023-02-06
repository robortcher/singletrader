# -*-coding:utf-8 -*-

import psycopg2
from psycopg2 import OperationalError
from datetime import datetime, date
import pandas as pd
import re
import numpy as np
import yaml
import os
import json
# from datacenter_utils.logger import logger
from singletrader.shared.logging import logger
from io import StringIO
import uuid


def read_yaml(file_path, key='pg'):
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg[key]


class Postgres(object):

    def __init__(self, conf=None, conf_path='/etc/db.yaml'):
        """
        生成数据库连接self.cursor。默认加载conf配置，当conf配置不存在时，再加载环境变量PG_CONF的配置，
        若PG_CONF的配置也不存在，最后加载conf_path（默认/etc/db.yaml）的文件配置
        优先级：conf字典配置 > 环境变量配置 > conf_path文件配置

        :param conf: 对应数据库配置
        :param conf_path: yaml配置文件绝对路径（可放在环境下任意路径），文件格式参考同目录下的db.yaml
        """
        if not conf:
            conf_str = os.environ.get('PG_CONF')
            if conf_str:
                conf = json.loads(conf_str)
                # conf = eval(conf_str)
                # if isinstance(conf, str):
                #     conf = eval(conf)
            if not conf:
                conf = read_yaml(conf_path)
        self.conf = conf
    
    @property
    def conn(self):
        return psycopg2.connect(database=self.conf.db_name,
                            user=self.conf.username,
                            password=self.conf.password,
                            host=self.conf.host,
                            port=int(self.conf.port))
        # try:
        #     self.database_postgres = psycopg2.connect(database=conf['database'],
        #                                               user=conf['user'],
        #                                               password=conf['password'],
        #                                               host=conf['host'],
        #                                               port=int(conf['port']))

        #     # self.cursor = self.database_postgres.cursor()
        # except OperationalError as e:
        #     raise e

    def is_table_exist(self, table_str):
        # TODO 表名字符长度限制63
        exec_str = "select exists(select * from information_schema.tables where table_name='{}');".format(
            table_str.lower()[:63])
        conf = self.conf
        conn = None
        try:
            conn = self.conn
            cursor = conn.cursor()
            cursor.execute(exec_str)
            exist = cursor.fetchone()[0]
            cursor.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            conn.rollback()
            exist = False
        finally:
            if conn:
                conn.close()
        logger.info('{} is_table_exist: {}'.format(table_str, exist))
        return exist

    def upsert_df(self, df, table_name, constraint_columns,
                  date_columns=None, timestamp_columns=None
                  ):
        """
        dataframe按照规定的字段设计upsert到指定数据库。
        针对constraint_columns列表，df任一记录任一字段与数据库不一样，就会插入一条新的数据，否则会更新数据库里原有记录。

        :param df: 待插入到数据库里的dataframe
        :type df: pandas.Dataframe
        :param table_name: 目标表名
        :type table_name: str
        :param constraint_columns: 表结构设计中的primary key
        :type constraint_columns: list of str
        :param date_columns: 定义为date类型的字段
        :type date_columns: list of str
        :param timestamp_columns:  定义为timestamp类型的字段
        :type timestamp_columns: list of str
        """
        conf = self.conf
        conn = self.conn
        # conn = psycopg2.connect(database=conf.db_name,
        #                             user=conf.username,
        #                             password=conf.password,
        #                             host=conf.host,
        #                             port=int(conf.port))
        cursor = conn.cursor()
        data_list = df.values.tolist()
        columns = list(df.columns)
        column_list = ['"{}"'.format(c) for c in columns]
        exclude_columns = column_list.copy()
        constraint_columns = ['"{}"'.format(i) for i in constraint_columns]

        for remove_c in constraint_columns:
            exclude_columns.remove(remove_c)
        for row in data_list:
            if len(column_list) - len(constraint_columns) == 1:
                sql = '''
                           INSERT INTO {t} 
                                VALUES  ({column_str})
                                ON CONFLICT ({constraint_columns})
                                DO UPDATE SET
                                    {exclude_columns}
                                    = {exclude_values} ;
                        '''
            elif len(column_list) - len(constraint_columns) > 1:
                sql = '''
                           INSERT INTO {t} 
                                VALUES  ({column_str})
                                ON CONFLICT ({constraint_columns})
                                DO UPDATE SET
                                    ({exclude_columns})
                                    = ({exclude_values}) ;
                        '''
            else:
                raise Exception('length of constraint_columns error')

            for i in range(len(row)):
                if date_columns and column_list[i] in date_columns:
                    if isinstance(row[i], pd.Timestamp):
                        row[i] = row[i].date()
                    elif isinstance(row[i], str):
                        row[i] = datetime.strptime(row[i], '%Y-%m-%d').date()
                    elif isinstance(row[i], date):
                        pass
                    elif isinstance(row[i], datetime):
                        row[i] = row[i].date()
                    else:
                        row[i] = None

                    if timestamp_columns and column_list[i] in timestamp_columns:
                        if isinstance(row[i], pd.Timestamp):
                            row[i] = row[i].datetime()
                        elif isinstance(row[i], str):
                            row[i] = datetime.strptime(
                                row[i], '%Y-%m-%d %H:%M:%S')
                        elif isinstance(row[i], date):
                            pass
                        elif isinstance(row[i], datetime):
                            pass
                        else:
                            row[i] = None
                if row[i] == '' or (isinstance(row[i], float) and np.isnan(row[i])):
                    row[i] = None

            c_str = ','.join(['%s' for i in range(len(row))])

            exclude_values = ['EXCLUDED.{}'.format(i) for i in exclude_columns]
            sql_format = sql.format(t=table_name,
                                    column_str=c_str,
                                    constraint_columns=','.join(
                                        constraint_columns),
                                    exclude_columns=','.join(exclude_columns),
                                    exclude_values=','.join(exclude_values))
            cursor.execute(sql_format, row)

        logger.info('saved {}'.format(table_name))
        conn.commit()
        cursor.close()
        conn.close()

    def update_insert_df(self, df, table_name, text_columns, constraint_columns,
                         date_columns=None, timestamp_columns=None
                         ):
        """
        dataframe按照规定的字段设计upsert到指定数据库。
        针对constraint_columns列表，df任一记录任一字段与数据库不一样，就会插入一条新的数据，否则会更新数据库里原有记录。
        当表（table_name）不存在时，该函数操作会自动创建表，按照字段类型参数（text_columns、constraint_columns、date_columns、
        timestamp_columns）创建相应类型，未指定的按照float类型处理

        :param df: 待插入到数据库里的dataframe，其columns需要明确且与text_columns、constraint_columns、date_columns、timestamp_columns保持一致
        :type df: pandas.Dataframe
        :param table_name: 目标表名
        :type table_name: str
        :param text_columns: 定义为text类型的字段
        :type text_columns: list of str
        :param constraint_columns: 设置为CONSTRAINT
        :type constraint_columns: list of str
        :param date_columns: 定义为date类型的字段
        :type date_columns: list of str
        :param timestamp_columns:  定义为timestamp类型的字段
        :type timestamp_columns: list of str
        """
        conf = self.conf
        conn = self.conn
        # conn = psycopg2.connect(database=conf.db_name,
        #                             user=conf.username,
        #                             password=conf.password,
        #                             host=conf.host,
        #                             port=int(conf.port))
        cursor = conn.cursor()
        data_list = df.values.tolist()
        columns = list(df.columns)
        data_list.insert(0, columns)
        if date_columns:
            text_columns.extend(date_columns)
        if timestamp_columns:
            text_columns.extend(timestamp_columns)
        count = 0
        column_list = []
        c_str = ''
        for row in data_list:
            if count == 0:
                # 标题format
                column_list = [self._format_str(r) for r in row]

                if not text_columns:
                    text_columns = column_list
                sql_str = ','.join(['"{}" float'.format(j.replace(" ", "_")) if j not in text_columns
                                    else '"{}" text'.format(j.replace(" ", "_"))
                                    for j in column_list])
                if not date_columns:
                    date_columns = []
                for dd in date_columns:
                    sql_str = sql_str.replace(
                        '"{}" text'.format(dd), '"{}" date'.format(dd))

                if not timestamp_columns:
                    timestamp_columns = []
                for dd in timestamp_columns:
                    sql_str = sql_str.replace('"{}" text'.format(
                        dd), '"{}" timestamp'.format(dd))

                # sql_s = re.findall(r'[^\*/:?\\|<>!]', sql_s, re.S)
                # sql_s = "".join(sql_s)
                logger.info(sql_str)
                # 自定义数据类型
                if not self.is_table_exist(table_name):
                    add_conflict_sql = '''
                    ALTER TABLE "public"."{t}" ADD CONSTRAINT "{pkey}" PRIMARY KEY ({keys});
                    '''.format(t=table_name, pkey=table_name + '_pkey',
                               keys=','.join(['"{}"'.format(i)
                                             for i in constraint_columns])
                               )
                    cursor.execute(
                        'create table {} ({});'.format(table_name, sql_str))
                    cursor.execute(add_conflict_sql)
                    logger.info('create table: %s' % sql_str)
            else:
                if len(column_list) - len(constraint_columns) == 1:
                    sql = '''
                               INSERT INTO {t} 
                                    VALUES  ({column_str})
                                    ON CONFLICT ({constraint_columns})
                                    DO UPDATE SET
                                        {exclude_columns}
                                        = {exclude_values} ;
                            '''
                elif len(column_list) - len(constraint_columns) > 1:
                    sql = '''
                               INSERT INTO {t} 
                                    VALUES  ({column_str})
                                    ON CONFLICT ({constraint_columns})
                                    DO UPDATE SET
                                        ({exclude_columns})
                                        = ({exclude_values}) ;
                            '''
                else:
                    sql = ''''''

                for i in range(len(row)):
                    if date_columns and column_list[i] in date_columns:
                        if isinstance(row[i], pd.Timestamp):
                            row[i] = row[i].date()
                        elif isinstance(row[i], str):
                            row[i] = datetime.strptime(
                                row[i], '%Y-%m-%d').date()
                        elif isinstance(row[i], date):
                            pass
                        elif isinstance(row[i], datetime):
                            row[i] = row[i].date()
                        else:
                            row[i] = None

                        if timestamp_columns and column_list[i] in timestamp_columns:
                            if isinstance(row[i], pd.Timestamp):
                                row[i] = row[i].datetime()
                            elif isinstance(row[i], str):
                                row[i] = datetime.strptime(
                                    row[i], '%Y-%m-%d %H:%M:%S')
                            elif isinstance(row[i], date):
                                pass
                            elif isinstance(row[i], datetime):
                                pass
                            else:
                                row[i] = None
                    if row[i] == '' or (isinstance(row[i], float) and np.isnan(row[i])):
                        row[i] = None
                    # if filter_dict and column_list[i] in filter_dict.keys():
                    #     if filter_dict[column_list[i]] == 'text_format':
                    #         row_s = str(row[i])
                    #         row_s = re.findall(r'[^\*/:?\\|<>!\-]', row_s, re.S)
                    #         row[i] = "".join(row_s)

                # one_sql_str = ','.join(row)
                if not c_str:
                    c_str = ','.join(['%s' for i in range(len(row))])
                    exclude_list = column_list.copy()
                    for remove_c in constraint_columns:
                        exclude_list.remove(remove_c)
                    exclude_columns = ['"{}"'.format(i) for i in exclude_list]
                    constraint_columns = ['"{}"'.format(
                        i) for i in constraint_columns]
                    exclude_values = ['EXCLUDED."{}"'.format(
                        i) for i in exclude_list]
                    sql_str = sql_str.replace('text', '').replace(
                        'date', '').replace('float', '')
                sql_format = sql.format(t=table_name,
                                        column_str=c_str,
                                        constraint_columns=','.join(
                                            constraint_columns),
                                        exclude_columns=','.join(
                                            exclude_columns),
                                        exclude_values=','.join(exclude_values))

                cursor.execute(sql_format, row)
            count += 1

        logger.info('saved {}'.format(table_name))
        conn.commit()
        cursor.close()
        conn.close()

    def find(self, table, filter_dict=None, columns=None):
        """
        查询所有满足条件的结果

        :param table: 表格名称
        :param filter_dict: 筛选字典
        :param columns: 返回字段
        :type table: str
        :type filter_dict: dict
        :type columns: list of str
        :return: 查询结果
        """
        conf = self.conf
        conn = self.conn
        # conn = psycopg2.connect(database=conf['database'],
        #                         user=conf['user'],
        #                         password=conf['password'],
        #                         host=conf['host'],
        #                         port=int(conf['port']))
        cursor = conn.cursor()
        if columns:
            columns_str = ",".join(['"{}"'.format(i) for i in columns])
        else:
            columns_str = '*'
        filter_list = []
        if not filter_dict:
            filter_dict = {}
        for k, v in filter_dict.items():
            if isinstance(v, str):
                v = "'{}'".format(v)
            elif isinstance(v, (datetime, date)):
                v = "'{}'".format(v)
            filter_list.append('"{}" = {}'.format(k, v))
        filter_str = 'where' + ' and '.join(filter_list) if filter_list else ''
        sql = "select {} from {} {}".format(columns_str, table, filter_str)
        logger.info('sql: %s' % sql)
        cursor.execute(sql)
        res = cursor.fetchall()
        cursor.close()
        conn.close()
        return res

    def find_one(self, table, filter_dict, columns):
        """
        查询满足条件的一个结果

        :param table: 表格名称
        :param filter_dict: 筛选字典
        :param columns: 返回字段
        :type table: str
        :type filter_dict: dict
        :type columns: list of str
        :return: 查询结果
        """
        conf = self.conf
        conn = self.conn
        # conn = psycopg2.connect(database=conf['database'],
        #                         user=conf['user'],
        #                         password=conf['password'],
        #                         host=conf['host'],
        #                         port=int(conf['port']))
        cursor = conn.cursor()
        if columns:
            columns_str = ",".join(['"{}"'.format(i) for i in columns])
        else:
            columns_str = '*'
        filter_list = []
        if not filter_dict:
            filter_dict = {}
        for k, v in filter_dict.items():
            if isinstance(v, str):
                v = "'{}'".format(v)
            elif isinstance(v, (datetime, date)):
                v = "'{}'".format(v)
            filter_list.append('"{}" = {}'.format(k, v))
        filter_str = 'where' + ' and '.join(filter_list) if filter_list else ''
        sql = "select {} from {} {}".format(columns_str, table, filter_str)
        logger.info('sql: %s' % sql)
        cursor.execute(sql)
        res = cursor.fetchone()
        cursor.close()
        conn.close()
        return res

    def find_by_sql(self, sql):
        """
        通过sql语句查询，返回所有结果
        :param sql: 查询语句
        :return:
        """
        conf = self.conf
        conn = self.conn
        # conn = psycopg2.connect(database=conf['database'],
        #                         user=conf['user'],
        #                         password=conf['password'],
        #                         host=conf['host'],
        #                         port=int(conf['port']))
        cursor = conn.cursor(f'{uuid.uuid1()}')
        cursor.execute(sql)
        # count = 0
        # exit()
        res = []
        while True:
            r = cursor.fetchmany(180)
            if not r:
                break
            # count += cursor.rownumber
            # logger.info(f'{count} rows fetch completed')
            res.extend(r)
            
        cursor.close()
        conn.close()
        return res

    def close_connect(self):
        pass

    @staticmethod
    def _format_str(line):
        # 指定字符替换为 -
        line1 = re.sub(r"[\-\n\s \t]", "_", line)
        # 指定字符去除
        line2 = re.sub(r"[‘’“”，。？?]", "", line1)
        return line2

    def over_insert_df_pro(self, df, table_name, text_column, date_columns=None, add_update_time=True):
        conf = self.conf
        conn = self.conn
        # conn = psycopg2.connect(database=conf['database'],
        #                         user=conf['user'],
        #                         password=conf['password'],
        #                         host=conf['host'],
        #                         port=int(conf['port']))
        cursor = conn.cursor()
        if add_update_time:
            df['update_time'] = datetime.now()
        cc = df.values.tolist()
        columns_raw = df.columns.tolist()
        columns = ['"{}"'.format(i) for i in columns_raw]
        str_list = []
        for c in cc:
            str_list.append(','.join([str(i) for i in c]))
        string = '\n'.join(str_list)
        f = StringIO()  # StringIO 结构类似文件，但是内容都在内存里面
        # 循环写入数据到内存里面， 里面每个字段用制表符\t 隔开，每一行用换行符\n 隔开
        f.write(string)
        f.seek(0)
        if not self.is_table_exist(table_name):
            sql_list = []
            for dd in columns_raw:
                if dd in date_columns:
                    ddd = '"{}" date'.format(dd)
                elif dd in text_column:
                    ddd = '"{}" text'.format(dd)
                elif dd == 'update_time':
                    ddd = '"update_time" timestamp'
                else:
                    ddd = '"{}" float'.format(dd)
                sql_list.append(ddd)
            sql_str = ','.join(sql_list)
            sql = 'create table {} ({});'.format(table_name, sql_str)
            cursor.execute(sql)
            logger.info(sql)
        cursor.copy_from(f, table_name,
                         columns=columns,
                         sep=',', null='', size=16384) 

        conn.commit()
        cursor.close()
        conn.close()
        logger.info('saved {}'.format(table_name))

def get_static_read():
    conf_str = os.environ.get('PG_STATIC_READ')
    if conf_str:
        conf = json.loads(conf_str)
        return conf
    else:
        raise Exception('env variable PG_STATIC_READ unset')

def get_quote_read():
    conf_str = os.environ.get('PG_QUOTE_READ')
    if conf_str:
        conf = json.loads(conf_str)
        return conf
    else:
        raise Exception('env variable PG_QUOTE_READ unset')
        