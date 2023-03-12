import pandas as pd


"""
#收益月偏度计算
skew = data.groupby('asset').apply(lambda x:x['close'].droplevel('asset').pct_change().resample('M').apply(lambda x:x.skew()))
skew = skew.stack().swaplevel(0,1)
skew.name = 'skew'

#年度最高价距离计算 1 - close/Max(high,252)
distance = data.groupby('asset').apply(lambda x:(1-x['close'] / x['high'].rolling(252).max()).droplevel('asset').resample('M').last())
distance =  distance.stack().swaplevel(0,1)
distance.name = 'distance'
"""

def bar_resample(data,frequency,symbol_level=1,fields=None):
    """bar降采样函数"""
    data_output = {}

    if fields is None:
        fields = data.columns.tolist()
    for _field in fields:
        if _field == 'open':
            data_output[_field] = data[_field].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).first())
        elif _field == 'high':
            data_output[_field] = data[_field].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).max())
        elif _field == 'low':
            data_output[_field] = data[_field].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).min())
        elif _field in ['volume','money']:
            data_output[_field] = data[_field].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).sum())
        else:      
            data_output[_field] = data[_field].groupby(level=symbol_level).apply(lambda x:x.droplevel(symbol_level).resample(frequency).last())
    
    data_output = pd.concat(data_output,axis=1).swaplevel(0,1)
    return data_output

#封装好的multi factor testing 类
class MultiFactorTesting():

    def __init__(self,bar_data, factor_data,price_type='close',add_shift=1,market_cap_col='market_cap',freq=12):
        """
        
        
        """
        self.bar_data = bar_data
        self.factor_data = factor_data
        self.price_type = price_type
        self.add_shift = add_shift
        self.all_factors = factor_data.columns.tolist()
        self.market_cap_col = market_cap_col
        self.freq = freq
    
    # def compute_forward_returns(self,**kwargs):
    #     price_type = kwargs.get('price_type',self.price_type)
    #     add_shift = kwargs.get('add_shift', self.add_shift)
    #     prices = self.bar_data[price_type].unstack()
    #     forward_returns = prices.pct_change().shift(-(1+add_shift)).stack()
    #     forward_returns.name = 'next_return'
    #     return forward_returns
    
    def compute_forward_returns(self,**kwargs):
        """forward return = close/open-1"""
        price_type = kwargs.get('price_type',self.price_type)
        add_shift = kwargs.get('add_shift', self.add_shift)
        prices_close = self.bar_data['close'].unstack()
        prices_open = self.bar_data['open'].unstack()
        forward_returns = (prices_close/prices_open-1).shift(-(1+add_shift)).stack()
        forward_returns.name = 'next_return'
        return forward_returns

    def get_clean_factor(self,quantiles=5,labels=None):
        if isinstance(quantiles,int):
            quantiles = {factor:quantiles for factor in self.all_factors}
        if labels is None:
            labels = {factor: list(range(1,quantiles[factor]+1)) for factor in self.all_factors}   
        
        clean_data = {factor:self.factor_data[factor].groupby(level=0).apply(lambda x:pd.qcut(x,quantiles[factor],labels=labels[factor])) for factor in self.all_factors}
        return pd.concat(clean_data,axis=1)
    
    def get_clean_factor_return(self,quantiles=5,labels=None,**kwargs):
        return_data = self.compute_forward_returns(**kwargs)
        clean_data = self.get_clean_factor(quantiles=quantiles,labels=labels)
        merge_data = pd.concat([return_data,clean_data],axis=1)
        return merge_data
    
    def compute_forward_bm_return(self,w_mkt=None,**kwargs):
        return_data = self.compute_forward_returns(**kwargs)
        if w_mkt is None:
            market_return = return_data.groupby(level=0).mean()
            market_return.name = 'mkt'
            return market_return
        else:
            w_mkt = w_mkt.reindex(return_data.index) 
            market_return = (return_data*w_mkt).groupby(level=0).sum()
            market_return.name = 'mkt'
            return market_return
    
    def get_mkt_weights_cap(self):
        return self.bar_data[self.market_cap_col].groupby(level=0).apply(lambda x:x/x.sum())
    
    def compute_cell_return_eq(self,quantiles=5, labels=None,**kwargs):
        """计算每个cell的收益"""
        clean_factor_return = self.get_clean_factor_return(quantiles=quantiles,labels=labels,kwargs=kwargs)
        return clean_factor_return.groupby(level=0).apply(lambda x:x.groupby(self.all_factors).mean()).unstack().unstack().droplevel(0,axis=1)
    

    def compute_cell_return_cap(self,quantiles=5,labels=None,**kwargs):
        clean_factor_return = self.get_clean_factor_return(quantiles=quantiles, labels=labels,kwargs=kwargs)
        market_cap = self.bar_data[self.market_cap_col]
        clean_factor_return_cap = pd.concat([clean_factor_return,market_cap],axis=1)
        cell_returns_cap = clean_factor_return_cap.groupby(level=0).apply(lambda x:x.groupby(self.all_factors).apply(lambda x:(x['next_return']*x[self.market_cap_col]/x[self.market_cap_col].sum()).sum()))
        return cell_returns_cap
    

    def summary(self,quantiles=5, labels=None,**kwargs):
        clean_factor_return = self.get_clean_factor_return(quantiles=quantiles,labels=labels,**kwargs)
        
        cell_returns_eq = self.compute_cell_return_eq(quantiles=quantiles,labels=labels,**kwargs)
        cell_returns_cap = self.compute_cell_return_cap(quantiles=quantiles,labels=labels,**kwargs)
        mkt_returns_eq = self.compute_forward_bm_return()
        w_mkt = self.bar_data[self.market_cap_col].groupby(level=0).apply(lambda x:x/x.sum())
        mkt_returns_cap = self.compute_forward_bm_return(w_mkt=w_mkt)

        #计算等权市场超额
        excess_return_eq = cell_returns_eq.apply(lambda x:x-mkt_returns_eq)
        #计算市值市场超额
        excess_return_cap = cell_returns_cap.apply(lambda x:x-mkt_returns_cap)

        
        mkt_avg_ann_return_eq = round(mkt_returns_eq.mean()*100*self.freq,2)
        mkt_avg_ann_return_cap = round(mkt_returns_cap.mean()*100*self.freq,2)

        summary = {}

        summary[f'avg_annual_excess_return_eq(%)_with_mkt={mkt_avg_ann_return_eq}%'] = round(excess_return_eq.mean().unstack().T * 100 *self.freq,2)
        # summary[f'avg_anual_excess_return_cap(%)_with_mkt={mkt_avg_ann_return_cap}%'] = round(excess_return_cap.mean().unstack()*100*self.freq,2)


        # summary['hit_rate_eq(%)'] = round(excess_return_eq.apply(lambda x:x[x>0].count() / x.count()).unstack().T*100,2)
        # summary['hit_rate_cap(%)'] = round(excess_return_cap.apply(lambda x:x[x>0].count() / x.count()).unstack()*100,2)


        clean_factor_return_copy = clean_factor_return.copy()
        clean_factor_return_copy['next_return'] = 1
        sort_count = clean_factor_return_copy.groupby(level=0).apply(lambda x:x.groupby(self.all_factors).count()).unstack().unstack().droplevel(0,axis=1)
        # summary['avg_number_of_stocks'] = round(sort_count.mean().unstack().T,2)
        # summary['avg_proportion(%)']  =round(sort_count.apply(lambda x:x/x.sum(),axis=1).mean().unstack().T*100,2)


        return summary
    



if __name__ == '__main__':
    #数据获取&处理
    from pathlib import Path
    file = Path(__file__)
    parent_path = file.parent
    data_file = parent_path.__str__() + '/' + r'price_and_factor_data.csv'
    data = pd.read_csv(data_file)
    data['date'] = pd.to_datetime(data['date']) #date字段需要datetime格式；
    data = data.set_index(['date','asset']) #必须为date,asset双重索引；
    data = data.dropna()
    #data.head()
    """
                            close    open    high     low     avg      volume  market_cap      skew  distance
    date       asset                                                                                          
    2010-01-31 000001.XSHE  848.17  958.39  960.73  805.17  858.33  24294088.0    634.5328  0.232038  0.188175
    2010-02-28 000001.XSHE  877.48  848.56  884.13  820.81  875.92  10662390.0    656.4637  0.886190  0.160121
    2010-03-31 000001.XSHE  906.80  877.48  948.62  866.54  908.75  15706730.0    678.3945  0.091379  0.132058
    2010-04-30 000001.XSHE  803.61  909.53  932.59  754.75  798.92  20433608.0    601.1979 -1.859854  0.230826
    2010-05-31 000001.XSHE  684.40  785.63  793.45  665.24  694.56  18574160.0    512.0124 -0.454606  0.344928
    """

    mft = MultiFactorTesting(bar_data=data.iloc[:,:-2],factor_data=data.iloc[:,-2:],add_shift=0)
    # summary = mft.summary(quantiles=3,labels={'skew':['Low','Medium','High'],'distance':['Near','Med','Far']})
    summary = mft.summary(quantiles=5)
    for i in summary:
        print(i,'\n',summary[i],'\n')