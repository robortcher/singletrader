import plotly.graph_objects as go
import plotly.express as px
from math import ceil
from plotly.subplots import make_subplots

def subplots(df_list=[],cols=1):
    _n = df_list.__len__()
    rows = ceil(_n / cols)
    fig = make_subplots(rows, cols=cols)
    
    for _i in range(_n):
        current_row = _i // cols
        current_col = _i % cols
        _df = df_list[_i]
        for col in _df.columns:
            x = _df.index.tolist()
            y = _df[col].values.tolist()
            fig.append_trace(go.Scatter(
                x=x,
                y=y,
            ), row=current_row, col=current_col)
    return fig    


if __name__ == '__main__':
    print()