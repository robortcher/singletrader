from singletrader.datasdk.qlib.base import MultiFactor
from singletrader.factorlib.factortesting import FactorEvaluation
from config import basic_config
from init import names_bar,fields_bar


fields,names = basic_config()

fields += fields_bar
names += names_bar



if __name__ == '__main__':
    mf = MultiFactor(field=fields,name=names,start_date='2010-01-01',end_date = '2022-12-31')
    data = mf._data
    fe = FactorEvaluation(data[names_bar],data[basic_config()[1]])
    summary = fe.get_summary(add_shift=0,start_date='2010-01-01',end_date='2022-12-31',base='close',groups=10)
    print