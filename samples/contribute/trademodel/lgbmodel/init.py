import os
from singletrader.shared.utility import check_and_mkdir

path = __file__

home_dir =  os.path.dirname(path)


model_dir = home_dir+ '/'+ 'model'
pred_dir = home_dir + '/' + 'predict'


check_and_mkdir(model_dir)
check_and_mkdir(pred_dir)

fields_bar = ['$close','$open','$high','$low','$avg','$volume','$circulating_market_cap','$money',]
names_bar = ['close','open','high','low','avg','volume','circulating_market_cap','money']





if __name__ == '__main__':
    pass