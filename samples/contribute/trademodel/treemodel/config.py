def basic_config(*args,**kwargs):
    fields = []
    names = []

    windows  = [1,2,5,10,20]

    # n日涨幅
    fields += ["$close / Ref($close,%d) - 1" %d for d in windows]
    names += ["daily_return%d" %d for d in windows]

    # 收益波幅
    fields += ["($close / Ref($close,1) - 1) / Std($close / Ref($close,1) - 1, %d)" %d for d in (20,60,120)]
    names += ["retvol%d" % d for  d in (20,60,120)]

    # 换手率
    fields += ["Sum($turnover_ratio,%d)" %d for d in windows]
    names += ["TO%d" %d for d in windows]

    # 换手比
    fields += ["$turnover_ratio / Mean($turnover_ratio,%d)" %d for d in (20,60,120)]
    names += ["TOR%d" %d for d in (20,60,120)]

    # # 收益立方
    # fields += ["$close / Ref($close, %d) ** 3" %d for d in (1,2,5)]
    # names += ["return_cube%d" % d for d in (1,2,5)]


    return fields,names


if __name__ == '__main__':
   _f, _n =  basic_config()
   print('==')