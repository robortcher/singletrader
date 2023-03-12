import jqdatasdk as jq
import os
import pandas as pd
from singletrader import QLIB_BIN_DATA_PATH
import qlib
from qlib.data import D

dir = QLIB_BIN_DATA_PATH + '/'+'instruments'
qlib.init(provider_uri= QLIB_BIN_DATA_PATH)

def dump_instruments():
    all_concepts = jq.get_concepts()
    end_date = '2050-12-31'
    for concept in all_concepts.index:
        cons = jq.get_concept_stocks(concept)
        df = pd.DataFrame(index=cons)
        df['start_date'] = all_concepts.loc[concept]['start_date']
        df['end_date'] = end_date
        path = dir +'/'+ all_concepts.loc[concept]['name']+'.txt'
        df.to_csv(path,sep='\t',header=False)
    #     print('==')
    # print('===')
    

def get_all_concepts():
    all_concepts = [i[:-4] for i in os.listdir(dir)]
    # all_concepts = 
    all_concepts.remove('all')
    return all_concepts
    # print(r'==')

def get_concept_cons(concept):
    qlib_instruments = D.instruments(concept)
    instruments =  D.list_instruments(instruments=qlib_instruments, as_list=True)
    return instruments


if __name__ == '__mian__':
    all_concepts = get_all_concepts()
    cons = get_concept_cons(all_concepts[0])



    print('===')
    # dump_instruments()