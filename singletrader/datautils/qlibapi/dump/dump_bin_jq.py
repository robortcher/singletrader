# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
用于将数据库数据生成到qlib.bin格式
"""

from dataclasses import dataclass
import logging
import os
import abc
import shutil
import traceback
from pathlib import Path
from typing import Iterable, List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from functools import partial
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from loguru import logger
from qlib.utils import fname_to_code, code_to_fname

from singletrader import QLIB_BIN_DATA_PATH
from singletrader.datautils.dataapi.jqapi import get_security_info,get_trade_days,jq_bar_api,ext_bardata_api_jq, ext_bardata_api2_jq


__all__ = ['DumpAll', 'Update','get_initial_fields']






APIS = {
    "jq_bar_api":jq_bar_api,
    "ext_bardata_api_jq":ext_bardata_api_jq,
    "ext_bardata_api2_jq":ext_bardata_api2_jq
}

__Ignorded_Fields__ = ["update_date","industry_name"]
__Initial_Fields__ = [field for field in np.sum([i.all_factors for i in APIS.values()]) if field not in __Ignorded_Fields__]

def get_initial_fields():
    qlib_fields = ["$"+field for field in __Initial_Fields__]
    return qlib_fields

last_datetime = pd.Timestamp(dt.date.today() - dt.timedelta(1))
last_date_str = last_datetime.strftime("%Y-%m-%d")

class DumpDataBase:
    INSTRUMENTS_START_FIELD = "start_date"
    INSTRUMENTS_END_FIELD = "end_date"
    CALENDARS_DIR_NAME = "calendars"
    FEATURES_DIR_NAME = "features"
    INSTRUMENTS_DIR_NAME = "instruments"
    DUMP_FILE_SUFFIX = ".bin"
    DAILY_FORMAT = "%Y-%m-%d"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"
    INSTRUMENTS_SEP = "\t"
    INSTRUMENTS_FILE_NAME = "all.txt"
    UPDATE_MODE = "update"
    ALL_MODE = "all"
    def __init__(
        self,
        qlib_dir: str = QLIB_BIN_DATA_PATH,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 1,
        date_field_name: str = "date",
        symbol_field_name: str = "code",
        exclude_fields: str = "",
        include_fields: str = "",
        is_index: bool = False,
        
        start_date: str = "2022-08-01",
        end_date: str = last_date_str,
        
        trade_date: str = None,
        universe: list = None,
        factor_fields: str = [],
        api:str = "bardata_api",
        
        limit_nums: int = None,
        region: str = 'cn',
    ):
        """

        Parameters
        ----------
        csv_path: str
            stock data path or directory
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        if isinstance(exclude_fields, str):
            exclude_fields = exclude_fields.split(",")
        if isinstance(include_fields, str):
            include_fields = include_fields.split(",")
        if isinstance(factor_fields, str):
            factor_fields = factor_fields.split(",")
        self._exclude_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, exclude_fields)))
        self._include_fields = tuple(filter(lambda x: len(x) > 0, map(str.strip, include_fields)))
        self.symbol_field_name = symbol_field_name
        self.limit_nums = limit_nums
        self.region = region
        self.is_index = is_index
        self.start_date = start_date
        self.end_date = end_date
        self.start_date_datetime = pd.to_datetime(start_date)
        self.end_date_datetime = pd.to_datetime(end_date)
        
        self.universe =  universe
        self.trade_date = trade_date
        self.factor_fields = factor_fields
        self.api = api
        self.instruments_list = self.universe
        if limit_nums is not None:
            self.instruments_list = self.instruments_list[: int(limit_nums)]
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.backup_dir = backup_dir if backup_dir is None else Path(backup_dir).expanduser()
        if backup_dir is not None:
            self._backup_qlib_dir(Path(backup_dir).expanduser())

        self.freq = freq
        self.calendar_format = self.DAILY_FORMAT if self.freq == "day" else self.HIGH_FREQ_FORMAT

        self.works = max_workers
        self.date_field_name = date_field_name

        self._calendars_dir = self.qlib_dir.joinpath(self.CALENDARS_DIR_NAME)
        self._features_dir = self.qlib_dir.joinpath(self.FEATURES_DIR_NAME)
        self._instruments_dir = self.qlib_dir.joinpath(self.INSTRUMENTS_DIR_NAME)

        self._calendars_list = []

        self._mode = self.ALL_MODE
        self._kwargs = {}

    def _backup_qlib_dir(self, target_dir: Path):
        shutil.copytree(str(self.qlib_dir.resolve()), str(target_dir.resolve()))

    def _format_datetime(self, datetime_d: [str, pd.Timestamp]):
        datetime_d = pd.Timestamp(datetime_d)
        return datetime_d.strftime(self.calendar_format)

    def _get_date(
        self, file_or_df: [Path, pd.DataFrame], *, is_begin_end: bool = False, as_set: bool = False
    ) -> Iterable[pd.Timestamp]:
        if not isinstance(file_or_df, pd.DataFrame):
            df = self._get_source_data(file_or_df)
        else:
            df = file_or_df
        if df.empty or self.date_field_name not in df.columns.tolist():
            _calendars = pd.Series(dtype=np.float32)
        else:
            _calendars = df[self.date_field_name]

        if is_begin_end and as_set:
            return (_calendars.min(), _calendars.max()), set(_calendars)
        elif is_begin_end:
            return _calendars.min(), _calendars.max()
        elif as_set:
            return set(_calendars)
        else:
            return _calendars.tolist()

    def _get_source_data(self, code: str) -> pd.DataFrame:

        api = APIS[self.api]
        if self.factor_fields.__len__() == 0:
            factor_list = api.all_factors
        else:
            factor_list = self.factor_fields
        
        if self.api == 'jq_bar_api':
            df = api.query(universe=[code],start_date=self.start_date,factor_list=factor_list,end_date=self.end_date, trade_date=self.trade_date).reset_index()
        elif self.api != 'jq_api':
            df = api.query(universe=[code],start_date=self.start_date,factor_list=factor_list,end_date=self.end_date, trade_date=self.trade_date).reset_index()
        else:
            df = api.query(universe=[code],start_date=self.start_date,end_date=self.end_date).reset_index()
        if df is None or df.empty:
            logger.warning(f"{code} data is None or empty")
            return
        df[self.date_field_name] = df[self.date_field_name].apply(lambda x: self._format_datetime(x)).astype(np.datetime64)
        df.drop_duplicates([self.date_field_name], inplace=True)
        return df

    # def get_symbol_from_file(self, file_path: Path) -> str:
    #     return fname_to_code(file_path.name[: -len(self.file_suffix)].strip().lower())

    def get_dump_fields(self, df_columns: Iterable[str]) -> Iterable[str]:
        return (
            self._include_fields
            if self._include_fields
            else set(df_columns) - set(self._exclude_fields)
            if self._exclude_fields
            else df_columns
        )

    @staticmethod
    def _read_calendars(calendar_path: Path) -> List[pd.Timestamp]:
        return sorted(
            map(
                pd.Timestamp,
                pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
            )
        )

    def _read_instruments(self, instrument_path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            instrument_path,
            sep=self.INSTRUMENTS_SEP,
            names=[
                self.symbol_field_name,
                self.INSTRUMENTS_START_FIELD,
                self.INSTRUMENTS_END_FIELD,
            ],
        )

        return df

    def _load_instruments(self):
        instruments = get_security_info()
        instruments = instruments[instruments.type == 'stock']
        instruments = instruments[(instruments[self.INSTRUMENTS_END_FIELD] > self.start_date_datetime) & \
                                  (instruments[self.INSTRUMENTS_START_FIELD] <= self.end_date_datetime)]

        instruments.loc[self.end_date_datetime < instruments.end_date, 'end_date'] = self.end_date_datetime
        instruments_list = instruments[self.symbol_field_name].to_list()

        return instruments, instruments_list

    def save_calendars(self, calendars_data: list):
        calendars_path = str(self._calendars_dir.joinpath(f"{self.freq}.txt").expanduser().resolve())
        result_calendars_list = list(map(lambda x: self._format_datetime(x), calendars_data))
        self._calendars_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")
        # if self._mode == self.ALL_MODE:
        #     self._calendars_dir.mkdir(parents=True, exist_ok=True)
        #     np.savetxt(calendars_path, result_calendars_list, fmt="%s", encoding="utf-8")
        # elif self._mode == self.UPDATE_MODE:
        #     with open(calendars_path, 'a') as f:
        #         np.savetxt(f, result_calendars_list, fmt='%s', encoding='utf-8')

    def save_instruments(self, instruments_data: Union[list, pd.DataFrame]):
        self._instruments_dir.mkdir(parents=True, exist_ok=True)
        instruments_path = str(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME).resolve())
        if isinstance(instruments_data, pd.DataFrame):
            _df_fields = [self.symbol_field_name, self.INSTRUMENTS_START_FIELD, self.INSTRUMENTS_END_FIELD]
            instruments_data = instruments_data.loc[:, _df_fields]
            instruments_data[self.INSTRUMENTS_START_FIELD] = instruments_data[self.INSTRUMENTS_START_FIELD].apply(self._format_datetime)
            instruments_data[self.INSTRUMENTS_END_FIELD] = instruments_data[self.INSTRUMENTS_END_FIELD].apply(self._format_datetime)

            instruments_data.to_csv(instruments_path, header=False, sep=self.INSTRUMENTS_SEP, index=False)
        else:
            np.savetxt(instruments_path, instruments_data, fmt="%s", encoding="utf-8")

    def data_merge_calendar(self, df: pd.DataFrame, calendars_list: List[pd.Timestamp]) -> pd.DataFrame:
        # calendars
        calendars_df = pd.DataFrame(data=calendars_list, columns=[self.date_field_name])
        calendars_df[self.date_field_name] = calendars_df[self.date_field_name].astype(np.datetime64)
        cal_df = calendars_df[
            (calendars_df[self.date_field_name] >= df[self.date_field_name].min())
            & (calendars_df[self.date_field_name] <= df[self.date_field_name].max())
        ]
        # align index
        cal_df.set_index(self.date_field_name, inplace=True)
        df.set_index(self.date_field_name, inplace=True)
        r_df = df.reindex(cal_df.index)
        return r_df

    @staticmethod
    def get_datetime_index(df: pd.DataFrame, calendar_list: List[pd.Timestamp]) -> int:
        return calendar_list.index(df.index.min())

    def _data_to_bin(self, df: pd.DataFrame, calendar_list: List[pd.Timestamp], features_dir: Path):
        if df.empty:
            logger.warning(f"{features_dir.name} data is None or empty")
            return
        # align index
        _df = self.data_merge_calendar(df, calendar_list)
        date_index = self.get_datetime_index(_df, calendar_list)
        # used when creating a bin file
        for field in self.get_dump_fields(_df.columns):
            ignored = [self.symbol_field_name] + __Ignorded_Fields__
            if field in ignored:#(self.symbol_field_name, "update_date","industry_name"):
                continue
            bin_path = features_dir.joinpath(f"{field.lower()}.{self.freq}{self.DUMP_FILE_SUFFIX}")
            if field not in _df.columns:
                continue
            if bin_path.exists() and self._mode == self.UPDATE_MODE:
                # update
                with bin_path.open("ab") as fp:
                    np.array(_df[field]).astype("<f").tofile(fp)
            else:
                # append; self._mode == self.ALL_MODE or not bin_path.exists()
                if bin_path.exists():
                    os.remove(str(bin_path.resolve()))
                np.hstack([date_index, _df[field]]).astype("<f").tofile(str(bin_path.resolve()))

    def _dump_bin(self, code_or_data: [str, pd.DataFrame], calendar_list: List[pd.Timestamp]):
        if isinstance(code_or_data, pd.DataFrame):
            if code_or_data.empty:
                return
            code = fname_to_code(str(code_or_data.iloc[0][self.symbol_field_name]).lower())
            df = code_or_data
        elif isinstance(code_or_data, str):
            code = code_or_data
            df = self._get_source_data(code)
        else:
            raise ValueError(f"not support {type(code_or_data)}")
        if df is None or df.empty:
            self.instruments_list.remove(code)
            logger.warning(f"{code} data is None or empty")
            return

        # try to remove dup rows or it will cause exception when reindex.
        df = df.drop_duplicates(self.date_field_name)

        # features save dir
        features_dir = self._features_dir.joinpath(code_to_fname(code).lower())
        features_dir.mkdir(parents=True, exist_ok=True)
        self._data_to_bin(df, calendar_list, features_dir)

    @abc.abstractmethod
    def dump(self):
        raise NotImplementedError("dump not implemented!")

    def __call__(self, *args, **kwargs):
        self.dump()

class DumpDataAll(DumpDataBase):
    def _get_all_date(self):
        logger.info("start get all date......")
        all_datetime = set(get_trade_days(self.start_date, self.end_date))
        self._kwargs["all_datetime_set"] = all_datetime
        self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
        logger.info("end of get all date.\n")

    def _dump_calendars(self):
        logger.info("start dump calendars......")
        self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
        self.save_calendars(self._calendars_list)
        logger.info("end of calendars dump.\n")

    def _dump_instruments(self):
        logger.info("start dump instruments......")
        self.instruments, self.instruments_list = self._load_instruments()
        self.instruments = self.instruments[self.instruments[self.symbol_field_name].isin(self.instruments_list)]
        self.save_instruments(self.instruments)
        logger.info("end of instruments dump.\n")

    def _dump_features(self):
        logger.info("start dump features......")
        
        _dump_func = partial(self._dump_bin, calendar_list=self._calendars_list)
        if self.instruments_list is None:
            instruments = get_security_info(self.region)
            instruments = instruments[instruments.type == 'stock']
            instruments = instruments[(instruments[self.INSTRUMENTS_END_FIELD] > self.start_date_datetime) & \
                                    (instruments[self.INSTRUMENTS_START_FIELD] <= self.end_date_datetime)]

            instruments.loc[self.end_date_datetime < instruments.end_date, 'end_date'] = self.end_date_datetime
            self.instruments_list = instruments[self.symbol_field_name].to_list()
        with tqdm(total=len(self.instruments_list)) as p_bar:
            with ProcessPoolExecutor(max_workers=self.works) as executor:
                for _ in executor.map(_dump_func, self.instruments_list):
                    p_bar.update()

        logger.info("end of features dump.\n")

    def dump(self):
        self._get_all_date()
        if not self.is_index:
            self._dump_calendars()
            self._dump_instruments()
        else:
            self._calendars_list = sorted(map(pd.Timestamp, self._kwargs["all_datetime_set"]))
            self.instruments_list = self.universe
        self._dump_features()
        
class DumpDataUpdate(DumpDataBase):
    def __init__(
        self,
        qlib_dir: str = QLIB_BIN_DATA_PATH,
        backup_dir: str = None,
        freq: str = "day",
        max_workers: int = 16,
        date_field_name: str = "date",
        symbol_field_name: str = "code",
        exclude_fields: str = "",
        include_fields: str = "",
        limit_nums: int = None,
        universe:list = None,
        api:str = "bardata_api",
        is_index:bool = False,
        factor_fields: str = [],
        start_date = None,
        end_date = None
        
    ):
        """

        Parameters
        ----------
        qlib_dir: str
            qlib(dump) data director
        backup_dir: str, default None
            if backup_dir is not None, backup qlib_dir to backup_dir
        freq: str, default "day"
            transaction frequency
        max_workers: int, default None
            number of threads
        date_field_name: str, default "date"
            the name of the date field in the csv
        symbol_field_name: str, default "symbol"
            symbol field name
        include_fields: tuple
            dump fields
        exclude_fields: tuple
            fields not dumped
        limit_nums: int
            Use when debugging, default None
        """
        super().__init__(
            qlib_dir,
            backup_dir,
            freq,
            max_workers,
            date_field_name,
            symbol_field_name,
            exclude_fields,
            include_fields,
            api = api,
            is_index=is_index,
            universe=universe,
            factor_fields = factor_fields
        )
        self._mode = self.UPDATE_MODE
        self._old_calendar_list = self._read_calendars(self._calendars_dir.joinpath(f"{self.freq}.txt"))

        if start_date is None:
            self.start_date_datetime = self._old_calendar_list[-1] + pd.DateOffset(1)
        else:
            self.start_date_datetime = pd.to_datetime(start_date)
        
        if end_date is None:
            self.end_date_datetime = pd.Timestamp(dt.date.today() - dt.timedelta(1))
        else:
            self.end_date_datetime = pd.to_datetime(end_date)
        
        self.start_date = self.start_date_datetime.strftime("%Y-%m-%d")#self._old_calendar_list[-1] + pd.DateOffset(1)
        self.end_date = self.end_date_datetime.strftime("%Y-%m-%d")
        
        # NOTE: all.txt only exists once for each stock
        # NOTE: if a stock corresponds to multiple different time ranges, user need to modify self._update_instruments
        self._update_instruments = (
            self._read_instruments(self._instruments_dir.joinpath(self.INSTRUMENTS_FILE_NAME))
            .set_index([self.symbol_field_name])
            .to_dict(orient="index")
        )  # type: dict

        self.instruments, self.instruments_list = self._load_instruments()
        self._all_data = self._load_all_source_data()  # type: pd.DataFrame
        self._new_calendar_list = self._old_calendar_list + sorted(
            filter(lambda x: x > self._old_calendar_list[-1], self._all_data[self.date_field_name].unique())
        )

    def _load_all_source_data(self):
        # NOTE: Need more memory
        logger.info("start load all source data....")
        api = APIS[self.api]
        if self.factor_fields.__len__() == 0:
            factor_list = api.all_factors
        else:
            factor_list = self.factor_fields
        if self.api != 'jq_api':
            df = api.query(universe=self.instruments_list,start_date=self.start_date,factor_list=factor_list,end_date=self.end_date, trade_date=self.trade_date).reset_index()
        
        else:
            df = api.query(universe=self.instruments_list,start_date=self.start_date,end_date=self.end_date).reset_index()
            print('==debug')
        if df is None or df.empty:
            logger.warning(f"{self.instruments_list} data is None or empty")
            return
        df[self.date_field_name] = df[self.date_field_name].apply(lambda x: self._format_datetime(x)).astype(np.datetime64)
        df.drop_duplicates([self.date_field_name,self.symbol_field_name], inplace=True)
        logger.info("end of load all data.\n")
        return df

    def _dump_features(self):
        logger.info("start dump features......")
        error_code = {}
        with ProcessPoolExecutor(max_workers=self.works) as executor:
            futures = {}
            for _code, _df in self._all_data.groupby(self.symbol_field_name):
                _code = fname_to_code(str(_code).lower()).upper()
                _start, _end = self._get_date(_df, is_begin_end=True)
                if not (isinstance(_start, pd.Timestamp) and isinstance(_end, pd.Timestamp)):
                    continue
                if _code in self._update_instruments:
                    # exists stock, will append data
                    _update_calendars = (
                        _df[_df[self.date_field_name] > self._update_instruments[_code][self.INSTRUMENTS_START_FIELD]][
                            self.date_field_name
                        ]
                        .sort_values()
                        .to_list()
                    )
                    self._update_instruments[_code][self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                    futures[executor.submit(self._dump_bin, _df, _update_calendars)] = _code
                else:
                    # new stock
                    _dt_range = self._update_instruments.setdefault(_code, dict())
                    _dt_range[self.INSTRUMENTS_START_FIELD] = self._format_datetime(_start)
                    _dt_range[self.INSTRUMENTS_END_FIELD] = self._format_datetime(_end)
                    futures[executor.submit(self._dump_bin, _df, self._new_calendar_list)] = _code

            with tqdm(total=len(futures)) as p_bar:
                for _future in as_completed(futures):
                    try:
                        _future.result()
                    except Exception:
                        error_code[futures[_future]] = traceback.format_exc()
                    p_bar.update()
            logger.info(f"dump bin errors： {error_code}")

        logger.info("end of features dump.\n")

    def dump(self):
        self.save_calendars(self._new_calendar_list)
        self._dump_features()
        df = pd.DataFrame.from_dict(self._update_instruments, orient="index")
        df.index.names = [self.symbol_field_name]
        self.save_instruments(df.reset_index())

def dump_other_index_cons(start_date="2010-01-01",end_date=last_date_str,factor_list=['hs300','zz500','zz1000']):
    api = APIS["ext_bardata_api"]
    data = api.query(universe=None,start_date=start_date,end_date=end_date,factor_list=factor_list)
    data = data.unstack().fillna(0).stack().reset_index()
    def _get_start_end_date(data, label):
        start_date = data.loc[(data[label] > 0) & (data[label].shift(1, fill_value=0) == 0), 'date']
        end_date = data.loc[(data[label] > 0) & (data[label].shift(-1, fill_value=0) == 0), 'date']
        start_date.name = 'start_date'
        end_date.name = 'end_date'
        start_date.reset_index(inplace=True, drop=True)
        end_date.reset_index(inplace=True, drop=True)
        stock_start_end_date = pd.concat([start_date, end_date], axis=1)
        return stock_start_end_date
    for index in factor_list:
        _data = data[['code', 'date', index]]
        _index_list = _data.groupby('code').apply(lambda x: _get_start_end_date(x, index))
        _index_list = _index_list.droplevel(1)
        _max_date = _data['date'].max()
        _index_list.loc[_index_list['end_date'] == _max_date, 'end_date'] = '2022-06-30'
        _index_list.to_csv(f'{QLIB_BIN_DATA_PATH}/instruments/{index}.txt', sep='\t', header=False)
        logger.info(f"{index}成分股生成完毕...")
    logger.info(f"cons of {factor_list} dumped over...")   
    
def dump_other_industry_cons(start_date="2010-01-01",end_date=last_date_str):
    api = APIS["ext_bardata_api"]
    data = api.query(universe=None,start_date=start_date,end_date=end_date,factor_list=['industry_name'])
    data = pd.get_dummies(data,prefix=None)
    data.columns =  list(map(lambda x:x.lower(),data.columns))
    index_list = data.columns
    data = data.reset_index()
    # data = data.unstack().fillna(0).stack().reset_index()
    
    def _get_start_end_date(data, label):
        start_date = data.loc[(data[label] > 0) & (data[label].shift(1, fill_value=0) == 0), 'date']
        end_date = data.loc[(data[label] > 0) & (data[label].shift(-1, fill_value=0) == 0), 'date']
        start_date.name = 'start_date'
        end_date.name = 'end_date'
        start_date.reset_index(inplace=True, drop=True)
        end_date.reset_index(inplace=True, drop=True)
        stock_start_end_date = pd.concat([start_date, end_date], axis=1)
        return stock_start_end_date
    
    def _dump_index_cons(index):
        _data = data[['code', 'date', index]]
        index = _data.columns[-1]
        _index_list = _data.groupby('code').apply(lambda x: _get_start_end_date(x, index))
        _index_list = _index_list.droplevel(1)
        _max_date = _data['date'].max()
        _index_list.loc[_index_list['end_date'] == _max_date, 'end_date'] = '2022-06-30'
        _index_list.to_csv(f'{QLIB_BIN_DATA_PATH}/instruments/{index}.txt', sep='\t', header=False)
        logger.info(f"{index}成分股生成完毕...")

    
    parLapply(index_list, _dump_index_cons)

def DumpAll():


    DumpDataAll(api="jq_bar_api")()
    
    DumpDataAll(api="ext_bardata_api2_jq")()
    
    # DumpDataAll(api="ext_bardata_api_jq")()
   
    



def Update():

    # DumpDataUpdate(api="ext_bardata_api_jq")._dump_features()#聚宽备用股票拓展行情
    
    DumpDataUpdate(api="ext_bardata_api2_jq")._dump_features()#聚宽备用股票拓展行情2

    DumpDataUpdate(api='jq_bar_api')() #更新股票行情数据



def test():
    # DumpDataAll(api="jq_api",start_date='2018-01-01')._dump_features()
    # DumpDataAll(api="bardata_api")()
    # DumpDataUpdate(api="jq_api")._dump_features()#更新股票拓展行情2
    pass

if __name__ == "__main__":
    logging.info('nonono, nothing to say...')
    dump_other_industry_cons()