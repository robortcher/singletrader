from singletrader.processors.cs_processor import CsWinzorize,CsNeutrualize,Csqcut
from singletrader.shared.utility import save_pkl,load_pkl
from singletrader.factorlib import FactorEvaluation,summary_plot
import pandas as pd
import plotly.express as px
from plotly.figure_factory import create_table
import warnings
warnings.filterwarnings('ignore')
from singletrader.datasdk.qlib.base import MultiFactor
from singletrader.shared.utility import load_pkls
from singletrader.datasdk.sql.dataapi import get_index_cons
from singletrader.factorlib.factortesting import summary_plot

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
cs_neu = CsNeutrualize()
cs_qcut = Csqcut(q=5)
from config import get_feature_configs


from sklearn.ensemble import RandomForestClassifier


field_features,name_features = ['$high/Min($low,%d)-1' % d for d in (1,5,10)],['profit%d' %d for d in (1,5,10)]
field_features += ["$turnover_ratio * $turnover_ratio / Mean($turnover_ratio,%d)" %d for d in (1,5,10)]
name_features += ['TO%d' %d for d in (1,5,10)]

field_label = ['Ref($close,-%d-1)/Ref($close,-1)' % d for d in (1,2,5,10)]
name_label = ['ret%d' %d for d in (1,2,5,10)]


field_bar = ['$close','$avg','$open','$low','$volume','Sum(1-$paused,20)']
name_bar = ['close','avg','open','low','volume','used']


fields = field_bar + field_features + field_label
names = name_bar+name_features + name_label


if __name__ == '__main__':
    # mf = MultiFactor(field=fields,name=names,start_date='2015-01-01',end_date='2023-03-31')
    # data = mf._data.dropna()
    # data[name_label] =  cs_qcut(data[name_label])
    data = load_pkl(r'D:\projects\singletrader_pro\samples\factor_mining\tamplate01\data.pkl')
    
    X_sample = data[data.index.get_level_values(0)<'2023-01-01'][name_features]
    y_sample = data[data.index.get_level_values(0)<'2023-01-01']['ret5']

    

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    rf_clf = RandomForestClassifier(n_jobs=16)
    

    param_grid = {'max_depth': [1,2,5,10],
                'min_samples_split': [40,50,60],
                'min_samples_leaf': [10,15,20],
                'criterion':['entropy'],
                }    
    
    # dt_classifier = DecisionTreeClassifier(random_state=42)
    # model = GridSearchCV(dt_classifier, param_grid, n_jobs=16,cv=5)
    
    model = rf_clf
    model.fit(X_train, y_train)

    # print("Best parameters: ", model.best_params_)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy: {:.2f}".format(accuracy))

    print('===')



