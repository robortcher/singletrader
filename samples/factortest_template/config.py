from singletrader.factorlib.factortesting import FactorEvaluation
from singletrader.datasdk.qlib.base import MultiFactor
from singletrader.datasdk.sql.dataapi import get_index_cons
import numpy as np
import pandas as pd
import talib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from alphalens.tears import create_full_tear_sheet
def basic_config(*args,**kwargs):
    """create factors from config

    config = {
        'kbar': {}, # whether to use some hard-code kbar features
        'price': { # whether to use raw price features
            'windows': [0, 1, 2, 3, 4], # use price at n days ago
            'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
        },
        'volume': { # whether to use raw volume features
            'windows': [0, 1, 2, 3, 4], # use volume at n days ago
        },
        'rolling': { # whether to use rolling operator based features
            'windows': [5, 10, 20, 30, 60], # rolling windows size
            'include': ['ROC', 'MA', 'STD'], # rolling operator to use
            #if include is None we will use default operators
            'exclude': ['RANK'], # rolling operator not to use
        }
    }
    """
    fields = []
    names = []

    windows = [1,2,3,4,5,10,20]

    #动量
    fields += ["$close/Ref($close,%d)" % d for d in windows]
    names += ["mom%d" % d for d in windows]

    fields += ["Power(($close/Ref($close,%d)),2)" % d for d in windows]
    names += ["momsquare%d" % d for d in windows]

    #price loc
    fields += ["($close-Min($low,%d))/(Max($high,%d)-Min($low,%d))" % (d,d,d) for d in windows]
    names += ["loc%d" % d for d in windows]

    #换手
    fields += ["Mean($turnover_ratio,%d)" % d for d in windows]
    names +=["turnover%d" % d for d in windows]

    fields += ["$turnover_ratio/Mean($turnover_ratio, %d)" % d for d in windows]
    names += ["turnoverMA%d" % d for d in windows] 


    windows = [5,10,20,60]
    fields += ["Std($close, %d)/$close" % d for d in windows]
    names += ["STD%d" % d for d in windows]


    fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
    names += ["VSTD%d" % d for d in windows]

    fields += ["Corr(Log($close), Log($volume+1), %d)" % d for d in windows]
    names += ["CORR%d" % d for d in windows]

    fields += [
        "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
        for d in windows
    ]
    names += ["SUMP%d" % d for d in windows]

    
    fields += [
        "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
        for d in windows
    ]
    names += ["SUMN%d" % d for d in windows]


    fields += [
                "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                % (d, d)
                for d in windows
                ]
    names += ["WVMA%d" % d for d in windows]

    # fields += ["$pe_ratio/Mean($pe_ratio,%d)" % d for d in (20,60,120,250)]
    # names += ["PE%d" % d for d in (20,60,120,250)]



    fields += ["$circulating_market_cap","$market_cap","1/$pe_ratio","1/$pb_ratio","1/$pcf_ratio"]
    names += ["circ_mkt_cap","market_cap","ep","bp","cfp"]


    # fields += ["Log($circulating_market_cap)","$circulating_market_cap","$market_cap","1/$pe_ratio","1/$pb_ratio","1/$pcf_ratio"]
    # names += ["log_mkt_cap","circ_mkt_cap","market_cap","ep","bp","cfp"]


    return fields,names




# 加载股票历史数据
df = pd.read_csv('stock_data.csv')

# 定义一个自动挖掘因子的函数
def generate_factors(df):
    # 计算技术指标
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MA5'] = talib.MA(df['Close'], timeperiod=5)
    df['MA10'] = talib.MA(df['Close'], timeperiod=10)
    df['MA20'] = talib.MA(df['Close'], timeperiod=20)
    df['MA60'] = talib.MA(df['Close'], timeperiod=60)
    df['MACD'], _, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    
    # 计算因子
    df['Momentum'] = df['Close'] / df['Close'].shift(1) - 1
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Volume Change'] = df['Volume'] / df['Volume'].shift(1) - 1
    df['MA5-MA10'] = df['MA5'] / df['MA10'] - 1
    df['MA10-MA20'] = df['MA10'] / df['MA20'] - 1
    df['MA20-MA60'] = df['MA20'] / df['MA60'] - 1
    
    # 移除缺失值和无穷大值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # 返回包含所有因子的DataFrame
    return df

# 生成因子
df = generate_factors(df)

# 定义因子变量和标签变量
X = df.drop(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
y = df['Close'].pct_change()

# 定义特征选择器
selector = SelectKBest(score_func=f_regression, k=10)

# 定义标准化器
scaler = StandardScaler()

# 定义PCA变换器
pca = PCA(n_components=10)

# 定义管道
pipeline = make_pipeline(selector, scaler, pca)

# 将管道应用到数据
X_transformed = pipeline.fit_transform(X, y)

# 将PCA变换后的因子转化为DataFrame
df_factors = pd.DataFrame(X_transformed, columns=['Factor '+str(i+1) for i in range(X_transformed.shape[1])])

# 输出因子
print(df_factors)

# 生成Alpha分析报告
create_full_tear_sheet(factor_data=df_factors, price=close_prices, long_short=False, group_neutral=False, quantiles=None)


if __name__ == '__main__':
    bars = ['$open','$close','$low','$high']
    names = ['open','close','low','high']
    features, fnames = basic_config()

    fields = bars +features
    names  = names+fnames

    mf = MultiFactor(field=fields,name=names,start_date='2018-01-01',end_date='2023-03-31')

    data = mf._data


    fe = FactorEvaluation(bar_data=data[names],factor_data=data[fnames])

    ic = fe.get_factor_ic(next_n=5)

    print('-')

"""



"""