def basic_config(*args,**kwargs):

    fields = []
    names = []

    windows = [1,2,3,4,5,10,20]

    #动量
    fields += ["$close/Ref($close,%d)" % d for d in windows]
    names += ["mom%d" % d for d in windows]

    #d日隔日动量
    fields += ["Sum($open/Ref($close,1),%d)" %d for d in windows]
    names += ["overnightmom%d" %d for d in windows]

    #d日跨度
    fields += ["$high / Min($low,%d)" %d for d in windows]
    names += ["range%d" % d for d in windows]

    #ill_liquidity
    fields += ["Sum(Abs($close/Ref($close,1)-1),%d) / Sum($turnover_ratio,%d)" %(d,d) for d in windows]
    names += ["ill_lquidity%d" %d for  d in windows]


    #avg_ratio
    fields += ["($close-Sum($money,%d)/Sum($volume,%d)) / ($open-Sum($money,%d)/Sum($volume,%d))" %(d,d,d,d) for d in windows]
    names += ['avg_ratio%d' %d for d in windows]


    #d日收益平方
    fields += ["Power(($close/Ref($close,%d)-1),2)" % d for d in windows]
    names += ["momsquare%d" % d for d in windows]

    #d日收益立方
    fields += ["Power(($close/Ref($close,%d)-1),3)" % d for d in windows]
    names += ["momcube%d" % d for d in windows]


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


if __name__ == '__main__':
   _f, _n =  basic_config()
   print('==')