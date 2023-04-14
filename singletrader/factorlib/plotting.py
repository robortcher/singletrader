import plotly.express as px 
from plotly.figure_factory import create_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser
def summary_plot(report,title='',mode='note-book'):
    """
    """


    fig_list = []
    # ic_decay
    fig_ic_decay = px.bar(report['ic_decay'].mean(),title='ic decay',)
    fig_ic_decay.update_layout(yaxis_range=[-0.2,0.2])
    fig_list.append(fig_ic_decay)


    # ic-series
    ic_series = report['ic_series']
    ic_series_ma = ic_series.rolling(12).mean()
    fig_ic_series = make_subplots()
    index = ic_series_ma.index
    fig_ic_series.add_trace(
        go.Scatter(x=index ,y=ic_series_ma.values.tolist(),name='ic_ma12')
    )
    fig_ic_series.add_trace(
        go.Bar(x=index, y=ic_series.values.tolist(), name="ic")
    )
    fig_ic_series.update_layout(title='ic series')
    fig_list.append(fig_ic_series)


    # 分组净值
    group_nvs = report['group_nvs']
    fig_group_nvs = px.line(group_nvs-1)
    fig_group_nvs.update_layout(title='cumulative excess return(compound) of different groups' + '-' + group_nvs.name)
    # fig_group_nvs.show()
    fig_list.append(fig_group_nvs)

    # 多头收益
    group_return_long = report['group_return_long']
    group_return_long_ma = group_return_long.rolling(12).mean()
    fig_group_return_long = make_subplots()
    index = group_return_long_ma.index
    fig_group_return_long.add_trace(
        go.Scatter(x=index ,y=group_return_long_ma.values.tolist(),name='return_ma12')
    )
    fig_group_return_long.add_trace(
        go.Bar(x=index, y=group_return_long.values.tolist(), name="return")
    )
    fig_group_return_long.update_layout(title='long set return')
    # fig_group_return_long.show()
    fig_list.append(fig_group_return_long)


    # 空头收益
    group_return_short = report['group_return_short']
    group_return_short_ma = group_return_short.rolling(12).mean()
    fig_group_return_short = make_subplots()
    index = group_return_short_ma.index
    fig_group_return_short.add_trace(
        go.Scatter(x=index ,y=group_return_short_ma.values.tolist(),name='return_ma12')
    )
    fig_group_return_short.add_trace(
        go.Bar(x=index, y=group_return_short.values.tolist(), name="return")
    )
    fig_group_return_short.update_layout(title='short set return')
    # fig_group_return_short.show()
    fig_list.append(fig_group_return_short)

    
    fig_ret_SR_TO = make_subplots(rows=1,cols=3)
    # 分组年化收益
    ann_ret = report['ann_ret']
    index = ann_ret.index
    fig_ret_SR_TO.add_trace(
        go.Bar(x=index,y=ann_ret.values,name='ann_ret'),row=1,col=1
    )
    # fig_ann_ret.show()


    # 分组夏普
    SR = report['SR']
    index = SR.index
    fig_ret_SR_TO.add_trace(
        go.Bar(x=index,y=SR.values,name='SR'),row=1,col=2
    )
    # fig_ann_ret.s

    # 分组换手
    TO = report['TO']
    index = TO.index
    fig_ret_SR_TO.add_trace(
        go.Bar(x=index,y=TO.values,name='TO'),row=1,col=3
    )
    fig_ret_SR_TO.update_layout(title='ann_ret, SR and TO')
    fig_list.append(fig_ret_SR_TO)

    # 分组评估指标
    perfs = report['excess_performance']
    table_perfs = create_table(perfs.reset_index())
    fig_list.append(table_perfs)
    # for _fig in fig_list:
        
    #     with open(title.replace(' ',"_")+'.html', 'a') as f:    
    #         f.write(_fig.to_html(full_html=False, include_plotlyjs='cdn'))
    #     if show:
    #         _fig.show()
    for _i in range(len(fig_list)):
        _fig = fig_list[_i]
        if mode !='note-book':
            mode = 'a' if _i > 0 else 'w'
            file = title.replace(' ',"_")+'.html'
            with open(file, mode) as f:    
                f.write(_fig.to_html(full_html=False, include_plotlyjs='cdn'))
            
            webbrowser.open( file,new=2)
        else:
            _fig.show()



