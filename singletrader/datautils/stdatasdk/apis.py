
import singletrader
from singletrader.datautils.qlibapi.constructor.base import MultiFactor
from singletrader.datautils.qlibapi.constructor.base import __bar_fields__,__bar_names__
# from singletrader.datautils.qlibapi.dump.dump_concepts_cons import get_all_concepts,get_concept_cons
# except:
#     from qlibapi.constructor.base import MultiFactor
#     from qlibapi.constructor.base import __bar_fields__,__bar_names__
def get_price(instruments=None,start_date=None,end_date=None):
    mf = MultiFactor(instruments=instruments,field=__bar_fields__,name=__bar_names__,start_date=start_date,end_date=end_date)
    data = mf._data
    return data




# import pandas as pd
# def func(df,group_dict):
#     res = {key:df[df.index.get_level_values('instrument').isin(group_dict[key])].mean() for key in group_dict}
#     return pd.Series(res)

def summarize(group_dict,fields=['$money-Ref($money,5)'],start_date='2023-01-01',end_date=None):
    mf = MultiFactor(field=fields,name=['cash_flow_delta'],start_date=start_date,end_date=end_date)
    all_data = mf._data
    return all_data.groupby(level='datetime').apply(lambda x:func(x,group_dict=group_dict))
    
    print('--')


# def get_all_concepts():
#     all_concepts = [i[:-4] for i in os.listdir(dir)]
#     # all_concepts = 
#     all_concepts.remove('all')
#     return all_concepts
#     # print(r'==')

# def get_concept_cons(concept):
#     qlib_instruments = D.instruments(concept)
#     instruments =  D.list_instruments(instruments=qlib_instruments, as_list=True)
#     return instruments


# group_dict = {i:get_concept_cons(i) for i in get_all_concepts()}
# 
group_dict={'a':['000001.XSHE','600000.XSHG']}



d = summarize(group_dict)

print(d)