
from singletrader.datautils.qlibapi.constructor.base import MultiFactor
import pandas as pd
from singletrader.datautils.qlibapi.dump.dump_concepts_cons import get_all_concepts,get_concept_cons
from jqdatasdk import get_all_securities,query,finance


# def func(df,group_dict):
#     res = {key:df[df.index.get_level_values('instrument').isin(group_dict[key])].mean() for key in group_dict}
#     return pd.DataFrame(res).T


def func(df,group_dict):
    res = {key:df[df.index.get_level_values('instrument').isin(group_dict[key])].mean()  for key in group_dict}
    if isinstance(df,pd.Series):
        return pd.Series(res)
    else:
        return pd.DataFrame(res).T

def summarize(group_dict,fields=['$money-Ref($money,5)','Ref($close,-5)/$close-1'],start_date='2023-01-01',end_date=None):
    mf = MultiFactor(field=fields,name=fields,start_date=start_date,end_date=end_date)
    all_data = mf._data
    return all_data, all_data.groupby(level='datetime').apply(lambda x:func(x,group_dict=group_dict))
        
group_dict = {i:get_concept_cons(i) for i in get_all_concepts()}


if __name__ == '__main__':
    def get_all_manage_info(group_dict):
        universe = get_all_securities().index.tolist()
        for stk in universe:
            q=query(finance.STK_MANAGEMENT_INFO).filter(finance.STK_MANAGEMENT_INFO.code
                                                        )
            df=finance.run_query(q).rename(colume={'pub_date':'date'})

    print('--')
    d = summarize(group_dict,fields=['$money'])
    a,b = d
    a = a.unstack().T
    b = b.unstack()
    print(d)