# -*- coding: utf-8 -*-
"""
小工具开发
"""
import pandas as pd
import pickle
import time
import dask
from dask import compute,delayed
from multiprocessing import cpu_count
import functools
import os
CORE_NUM = cpu_count()

def load_pkl(filename):
    """载入pickle文件"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data 

def save_pkl(context, filename):
    """存储文件到pickle"""
    with open(filename, 'wb') as f:
        data = pickle.dump(context, f)
    return data 
#批量读取csv
def read_csvs(dire_path, **kwgs):
    """读取某一文件下的所有文件并转为DataFrame结构"""
    
    all_files = os.listdir(dire_path)
    all_files = [x for x in all_files if not x.startswith('.')]
    all_abs_files = list(map(lambda x: dire_path + '/' + x, all_files))
    datas = [pd.read_csv(file, **kwgs) for file in all_abs_files]
    datas_df = pd.concat(datas,axis=1)
    return datas_df

def load_pkls(dire_path=None,files=None,n=None):
    if (dire_path is None) and (files is None):
        raise Exception('dire_path and files must input one')
    if dire_path is not None:
        all_files = os.listdir(dire_path)
    elif files is not None:
        all_files = files
    all_files.sort()
    if n is not None:  
        all_files = all_files[:n]
    all_files = [x for x in all_files if not x.startswith('.')]
    if dire_path is not None:
        all_abs_files = list(map(lambda x: dire_path + '/' + x, all_files))
    else:
        all_abs_files = all_files
    datas = [load_pkl(file) for file in all_abs_files]
    return datas

def parLapply(iterable, func, CORE_NUM=CORE_NUM,*args, **kwargs):
    # start_time = time.time()
    with dask.config.set(scheduler = 'processes', num_workers = CORE_NUM):
        f_par = functools.partial(func, *args, **kwargs)
        result = compute([delayed(f_par)(item) for item in iterable])[0]
    # end_time = time.time()
    # deltahours = (end_time - start_time) / 3600
    # print(f'use time {deltahours} hours')
    return result

