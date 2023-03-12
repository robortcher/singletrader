import pandas as pd

def CSzscore(df,group_name='datetime'):
    return df.groupby(level=group_name).apply(lambda x:(x-x.mean()) / x.std())

