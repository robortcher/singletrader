from singletrader.factortesting.factor_testing import FactorEvaluation
from singletrader.datautils.qlibapi.constructor import MultiFactor
from samples.contribute.trademodel.configs import basic_config
from singletrader.shared.utility import save_pkl,load_pkl

start_date = '2010-01-01'
end_date = '2022-12-31'


# fields = []
# names = []






# features = ['Gap','turnover_ratio']
# fields += ['Ref($open,-1)/$close','Cov($turnover_ratio,$close/Ref($close,1),10)/Std($turnover_ratio,10)/Std($close/Ref($close,1),10)']
# names += features

fields,names = basic_config()
features = names

fields += ['$close','$open','$high','$low','$avg','$volume']
bars = ['close','open','high','low','avg','volume']
names += bars

marks =['paused','gt250']
names += marks
fields +=['$paused','Mean($close,250)']


if __name__ == '__main__':
    from singletrader.performance.common import performance_indicator
    ic = load_pkl(r'D:\projects\singletrader\ic-test-result2.pkl')
    features = basic_config()[1]
    ic = ic[features].swaplevel(0,1,axis=1)
    factor_effective = {}
    train_ic = ic[:'2020-12-31']
    for period in ["%dD" % d for d in range(1,11)]:
        icmean = train_ic[period].mean()
        factor_effective[period] = {}
        factor_effective[period]['postive'] = icmean[icmean>=0.03].index.tolist() 
        factor_effective[period]['negative'] = icmean[icmean<=-0.03].index.tolist()   



    from alphalens.utils import get_clean_factor_and_forward_returns
    from singletrader.factortesting.factor_testing import FactorEvaluation as FE
    mf = MultiFactor(field=fields,name=names,start_date='2015-01-01',end_date='2022-12-31')
    data = mf._data
    test_data = data.swaplevel(0,1)
    test_data.index = test_data.index.set_names(['date','asset'])
    # test_data = test_data[~test_data['gt250'].isna()]
    fe = FactorEvaluation(bar_data=test_data[bars],factor_data=test_data[features].dropna())

    # train_ic = ic[:'2021-01-01']

    # ics = fe.get_factor_ics(periods=(1,2,3,4,5,6,7,8,9,10),base='avg',total=False)

