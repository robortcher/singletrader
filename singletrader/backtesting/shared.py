import pandas as pd
import numpy as np

class ArrayManger(object):
    def __init__(self,size=100,universe=[],cols=[],expanding=False):
        """
        用于记录的过去一段时间窗口数据的矩阵管理器
        size:int, 矩阵管理器的长度
        universe:list, 涉及到的股票池，用作管理器的列
        cols:list,  涉及到的字段
        """
        self.count = 0
        self.size = size
        self.inited = False
        for _col in cols:
            setattr(self,_col,pd.DataFrame(np.nan,index=np.arange(self.size), columns=universe))
        self.cols = cols
        self.expanding = expanding

    def update_bar(self,bar):
        for _col in self.cols:
            _df = getattr(self,_col)
            if self.inited and self.expanding:
                _df.loc[self.count] = getattr(bar,_col).reindex(_df.columns).values
 
            
            else:
                _df.iloc[:-1] = _df.iloc[1:]
                _df.iloc[-1, :] = getattr(bar,_col).reindex(_df.columns).values
            setattr(self,_col,_df)
        self.count += 1
        if not self.inited and self.count >=  self.size:
            self.inited = True
            
