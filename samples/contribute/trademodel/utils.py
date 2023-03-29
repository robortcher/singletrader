import math
def get_period_split(start_year = 2010,
    min_train_year = 5,
    fix_train_year = 5,
    valid_period = 2,
    end_year = 2023,
    round_period = 1,
  ):
    #区间划分

    if fix_train_year is None:
        train_round = math.ceil((end_year-valid_period - start_year - min_train_year) /round_period)
        trains = [("%d-01-01"%(start_year),"%d-12-31"%(start_year+i+min_train_year)) for i in range(train_round)]
        valids = [("%d-01-01"%(start_year+i+min_train_year+1),"%d-12-31"%(start_year+i+min_train_year+valid_period)) for i in range(train_round)]
        tests = [("%d-01-01"%(start_year+i+min_train_year+valid_period+round_period),"%d-12-31"%(start_year+i+min_train_year+valid_period+round_period)) for i in range(train_round)]


    else:
        train_round = math.ceil((end_year-valid_period - start_year - fix_train_year) /round_period)
        trains = [("%d-01-01"%(start_year+i),"%d-12-31"%(start_year+i+fix_train_year)) for i in range(train_round)]
        valids = [("%d-01-01"%(start_year+i+fix_train_year+1),"%d-12-31"%(start_year+i+fix_train_year+valid_period)) for i in range(train_round)]
        tests = [("%d-01-01"%(start_year+i+fix_train_year+valid_period+round_period),"%d-12-31"%(start_year+i+fix_train_year+valid_period+round_period)) for i in range(train_round)]
    return trains,valids,tests



if __name__ == '__main__':
    p  = get_period_split()
    print('===')