from lightgbm import LGBMRegressor


lgb = LGBMRegressor()

def get_feature_configs(*args,**kwargs):
    fields = []
    names = []


    # 常用周期(短期窗口)
    windows = [1,2,3,4,5,10,20]

    # 短期动量(一阶)
    fields += ["$close/Ref($close,%d) - 1" % d for d in windows]
    names += ["mom%d" % d for d in windows]

    # 标准化收益(TS)
    fields += ["($close/Ref($close,1)-1) / (Std($close/Ref($close,1)-1,%d) + 1e-5)" %d for d in (5,10,20,60)]
    names += ["zret%d" % d for d in (5,10,20,60)]

    # 收益平方
    fields += ["Power(($close/Ref($close,%d)-1),2)" % d for d in windows]
    names += ["momsquare%d" % d for d in windows]

    # 当前n日最大收益
    fields += ["($close-Min($low,%d))/Min($low,%d)-1" % (d,d) for d in windows]
    names += ["maxprofit%d" %d for d in windows]

    # 当前n日最大亏损
    fields += ["(Max($high,%d)-$close)/Max($high,%d)-1" % (d,d) for d in windows]
    names += ["maxloss%d" %d for d in windows]

    # n日平均换手
    fields += ["Mean($turnover_ratio,%d)" % d for d in windows]
    names +=["turnover%d" % d for d in windows]

    # n日单位换手金额
    fields += ["Mean($money/$turnover_ratio,%d)" % d for d in windows]
    names += ["MOTO%d" % d for d in windows]

    # 换手乖离率
    fields += ["$turnover_ratio/Mean($turnover_ratio, %d)" % d for d in windows[1:]]
    names += ["turnoverMA%d" % d for d in windows[1:]] 

    # Gap
    fields += ["Sum($open/Ref($close,1)-1,%d)" %d for d in [1,5,10]]
    names += ["Gap%d" %d for d in [1,5,10]]

    windows = [5,10,20,60] # 中期常用窗口
    
    # 收益波动率
    fields += ["Std($close/Ref($close,1)-1, %d)" % d for d in windows]
    names += ["STD%d" % d for d in windows]

    # 风险调整收益
    fields += ["Mean($close/Ref($close,1)-1, %d) / Std($close/Ref($close,1)-1, %d)" % (d,d) for d in windows]
    names += ["sharpe%d" % d for d in windows]

    
    # 成交量波动率
    fields += ["Std($volume, %d)/Mean($volume,%d)" % (d,d) for d in windows]
    names += ["VSTD%d" % d for d in windows]

    # 价量相关性
    fields += ["Corr($close/Ref($close,1)-1, $volume, %d)" % d for d in windows]
    names += ["CORR%d" % d for d in windows]


    # 价量相关
    fields += ['Slope(Log($close),%d) / Slope($turnover_ratio,%d)' % (d,d) for d in (5,10,20)]
    names += ['PTO%d' %d for d in (5,10,20)]

    # 收益效率比
    fields += [
        "Sum(Greater($close/Ref($close, 1)-1, 0), %d)/(Sum(Abs($close/Ref($close, 1)-1), %d)+1e-12)" % (d, d)
        for d in windows
    ]
    names += ["SUMP%d" % d for d in windows]

    # 损失效率比
    fields += [
        "Sum(Greater((Ref($close, 1)-$close)/Ref($close, 1), 0), %d)/(Sum(Abs($close/Ref($close, 1)-1), %d)+1e-12)" % (d, d)
        for d in windows
    ]
    names += ["SUMN%d" % d for d in windows]

    # pv
    fields += ["($close/Ref($close,%d)-1) / ($turnover_ratio / Mean($turnover_ratio,%d))" % (d,d) for d in windows]
    names += ["pv%d" %d for d in windows]


    # 基本面因子
    fields += ["Log($circulating_market_cap)","1/$pe_ratio","1/$pb_ratio"]
    names += ["mkt_cap","ep","bp"]

    fields_label = ['Ref($close,-5)/Ref($open,-1)-1', 'Ref($close,-2)/Ref($close,-1)','Ref($avg,-6) / Ref($avg,-1)']
    names_label = ['co5','cc1','aa5']

    fields_mark = ['$open','$close','$circulating_market_cap','Sum($paused,60)']
    names_mark = ['open','close','circulating_market_cap','paused']

    fields_all = fields + fields_label + fields_mark
    names_all = names + names_label + names_mark
    return fields_all,names_all,names,names_label,names_mark

if __name__ == '__main__':
    fields,names,feature,label,mark = get_feature_configs()
    from singletrader.datasdk.qlib.base import MultiFactor
    from singletrader.shared.utility import load_pkl,save_pkl
    from singletrader.processors.cs_processor import CsWinzorize,CsStandardize
    from sklearn.model_selection import train_test_split
    import numpy as np
    from lightgbm import LGBMRegressor
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score
    
    # mf = MultiFactor(field=fields,name=names, start_date='2010-01-01',end_date='2023-03-31')
    from singletrader.factorlib.factortesting import FactorEvaluation
    from sklearn.ensemble import RandomForestRegressor
    # data = mf._data    
    # load_pkl(data,r'D:\projects\singletrader_pro\samples\factor_mining\tamplate01\data.pkl')
    data = load_pkl(r'D:\projects\singletrader_pro\samples\factor_mining\tamplate01\data.pkl')
    cs_win = CsWinzorize() # 
    cs_std = CsStandardize()
    # data_feature = data[feature+label][data['paused']==0]#去掉过去60天未上市或者停牌的股票
    data_feature = data[feature+label]
    data_feature = cs_win(data_feature)
    data_feature_std = cs_std(data_feature[feature+label])
    data_feature_std = data_feature_std.dropna()
    X,y = data_feature_std[feature],data_feature_std['aa5']
    
    rfg = RandomForestRegressor()

    lgb_cvs = cross_val_score(lgb,X,y,cv=10)
    rfg_cvs = cross_val_score(rfg,X,y,cv=10,n_jobs=-1)
    
    print("paused...")