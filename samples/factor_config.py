
from singletrader.datautils.qlibapi.constructor.base import MultiFactor
from singletrader.factortesting.factor_testing import FactorEvaluation
__bar_field__ = ['$open', '$close', '$high', '$low', '$volume', '$money', '$avg', '$high_limit', '$low_limit', '$pre_close', '$paused', '$factor']
__bar_name__ = ['open', 'close', 'high', 'low', 'volume', 'money', 'avg', 'high_limit', 'low_limit', 'pre_close', 'paused', 'factor']
class Factor():
    def __init__(self,factor_field,factor_name=None):
        if type(factor_field) is str:
            factor_field = [factor_field]
        if type(factor_name) is str:
            factor_name = [factor_name]
        elif factor_name is None:
            factor_name = factor_field

        self.factor_name = factor_name
        self.factor_field = factor_field

    def laod_data(self,start_date,end_date,instruments='all'):
        mf = MultiFactor(name=__bar_name__ + self.factor_name, field=__bar_field__ + self.factor_field,instruments=instruments,start_date=start_date,end_date=end_date)
        self.data = mf._data.swaplevel(0,1)
        self.FE = FactorEvaluation(self.data[__bar_name__],self.data[self.factor_name])




def parse_config_to_fields(config = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "AVG"],
            },
            "rolling": {},
        }):
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
    if "kbar" in config:
        fields += [
            "($close-$open)/$open",
            "($high-$low)/$open",
            "($close-$open)/($high-$low+1e-12)",
            "($high-Greater($open, $close))/$open",
            "($high-Greater($open, $close))/($high-$low+1e-12)",
            "(Less($open, $close)-$low)/$open",
            "(Less($open, $close)-$low)/($high-$low+1e-12)",
            "(2*$close-$high-$low)/$open",
            "(2*$close-$high-$low)/($high-$low+1e-12)",
        ]
        names += [
            "KMID",
            "KLEN",
            "KMID2",
            "KUP",
            "KUP2",
            "KLOW",
            "KLOW2",
            "KSFT",
            "KSFT2",
        ]
    if "price" in config:
        windows = config["price"].get("windows", range(5))
        feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "AVG"])
        for field in feature:
            field = field.lower()
            fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
            names += [field.upper() + str(d) for d in windows]
    if "volume" in config:
        windows = config["volume"].get("windows", range(5))
        fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
        names += ["VOLUME" + str(d) for d in windows]
    if "rolling" in config:
        windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
        include = config["rolling"].get("include", None)
        exclude = config["rolling"].get("exclude", [])
        # `exclude` in dataset config unnecessary filed
        # `include` in dataset config necessary field
        use = lambda x: x not in exclude and (include is None or x in include)
        if use("ROC"):
            fields += ["Ref($close, %d)/$close" % d for d in windows]
            names += ["ROC%d" % d for d in windows]
        if use("MA"):
            fields += ["Mean($close, %d)/$close" % d for d in windows]
            names += ["MA%d" % d for d in windows]
        if use("STD"):
            fields += ["Std($close, %d)/$close" % d for d in windows]
            names += ["STD%d" % d for d in windows]
        if use("BETA"):
            fields += ["Slope($close, %d)/$close" % d for d in windows]
            names += ["BETA%d" % d for d in windows]
        if use("RSQR"):
            fields += ["Rsquare($close, %d)" % d for d in windows]
            names += ["RSQR%d" % d for d in windows]
        if use("RESI"):
            fields += ["Resi($close, %d)/$close" % d for d in windows]
            names += ["RESI%d" % d for d in windows]
        if use("MAX"):
            fields += ["Max($high, %d)/$close" % d for d in windows]
            names += ["MAX%d" % d for d in windows]
        if use("LOW"):
            fields += ["Min($low, %d)/$close" % d for d in windows]
            names += ["MIN%d" % d for d in windows]
        if use("QTLU"):
            fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
            names += ["QTLU%d" % d for d in windows]
        if use("QTLD"):
            fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
            names += ["QTLD%d" % d for d in windows]
        if use("RANK"):
            fields += ["Rank($close, %d)" % d for d in windows]
            names += ["RANK%d" % d for d in windows]
        if use("RSV"):
            fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
            names += ["RSV%d" % d for d in windows]
        if use("IMAX"):
            fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
            names += ["IMAX%d" % d for d in windows]
        if use("IMIN"):
            fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
            names += ["IMIN%d" % d for d in windows]
        if use("IMXD"):
            fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
            names += ["IMXD%d" % d for d in windows]
        if use("CORR"):
            fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
            names += ["CORR%d" % d for d in windows]
        if use("CORD"):
            fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
            names += ["CORD%d" % d for d in windows]
        if use("CNTP"):
            fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
            names += ["CNTP%d" % d for d in windows]
        if use("CNTN"):
            fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
            names += ["CNTN%d" % d for d in windows]
        if use("CNTD"):
            fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
            names += ["CNTD%d" % d for d in windows]
        if use("SUMP"):
            fields += [
                "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                for d in windows
            ]
            names += ["SUMP%d" % d for d in windows]
        if use("SUMN"):
            fields += [
                "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                for d in windows
            ]
            names += ["SUMN%d" % d for d in windows]
        if use("SUMD"):
            fields += [
                "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                for d in windows
            ]
            names += ["SUMD%d" % d for d in windows]
        if use("VMA"):
            fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
            names += ["VMA%d" % d for d in windows]
        if use("VSTD"):
            fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
            names += ["VSTD%d" % d for d in windows]
        if use("WVMA"):
            fields += [
                "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                % (d, d)
                for d in windows
            ]
            names += ["WVMA%d" % d for d in windows]
        if use("VSUMP"):
            fields += [
                "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                % (d, d)
                for d in windows
            ]
            names += ["VSUMP%d" % d for d in windows]
        if use("VSUMN"):
            fields += [
                "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                % (d, d)
                for d in windows
            ]
            names += ["VSUMN%d" % d for d in windows]
        if use("VSUMD"):
            fields += [
                "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                for d in windows
            ]
            names += ["VSUMD%d" % d for d in windows]
    #动量
    fields += ["$avg/Ref($avg,%d) " % i for i in (1,5,20)]
    names += ["Mom%d" % i for i in (1,5,20)]

    #波动率
    fields +=["Std($close/Ref($close,1),%d)" % i for i in (10,20,60)]
    names += ["Std%d" % i for i in (10,20,60)]

    #成交额
    fields += ["Sum($money, %d)" % i for i in (1,5,20)]
    names += ["money%d" % i for i in (1,5,20)]


    #换手率
    fields += ["Sum($turnover_ratio,%d) " % i for i in (1,5,20)]
    names += ["turnover_ratio%d" % i for i in (1,5,20)]

    #换手率波动率
    fields +=["Std($turnover_ratio,%d)" % i for i in (10,20,60)]
    names += ["StdTr%d" % i for i in (10,20,60)]

    # # 低开
    # fields += ["Mean(Ref($open,-1)/$close-1,%d)" % i for i in (1,5,20)]
    # names += ["Gap%d" % i for i in (1,5,20)]

    #现金流入
    fields += ["Sum(($avg - Ref($avg,1)) * $volume, %d) / Mean($money,%d)" % (i,i) for i in (1,5,20)]
    names += ["CashIn%d" % i for i in (1,5,20)]


    # #单位金额波幅
    # fields += ["((Max($high,%d) - Min($low, %d)) / Ref($close, %d)) / Sum($money,%d)" % (i,i,i,i) for i in (1,5,20)]
    # names += ["RangePerM%d" % i for i in (1,5,20)]

    #单位金额波幅
    fields += ["Sum(($high-$low) / Ref($close,1)-1,%d)" % d for d in (1,5,20)]
    names += ["RangePerM%d" % i for i in (1,5,20)]

    #价格
    fields +=['$close / $factor']
    names += ['price']

    #价量相关
    fields += ["Corr($avg, $volume,%d)" %i for i in (10,20,60)]
    names += ["CrPtr%d" %i for i in (10,20,60)]


    #barra
    fields += ['$market_cap','1/$pb_ratio', '0.79*1/$pe_ratio+0.21*1/$pcf_ratio']
    names += ['size','bp','ep']
    # fields.remove(fields[names.index('AVG0')])
    # names.remove(names[names.index('AVG0')])
    return fields, names

fields,names = parse_config_to_fields()
# names += barra
features ={}
features['name'] = names
features['field'] = fields

# mf = MultiFactor(field=fields,name=names,start_date=last_trade_date_str,end_date=last_trade_date_str,instruments='all')
# mf._data.dropna().head()

    
if __name__ == '__main__':
    a = Factor(['Ref($open,-1)/$close'])

    a.laod_data(start_date='2022-01-01',end_date='2022-12-31')


    print('==')