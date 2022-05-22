from plotly.subplots import make_subplots
from math import ceil
import plotly.express as px
import plotly.figure_factory as ff


def make_box_subplots(df, ncols=2):
    nrows = ceil((len(df.columns)) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[column for column in df.columns],
        vertical_spacing=0.02,
    )

    for v in range(len(df.columns)):
        for t in px.box(df, y=df.columns[v]).data:
            fig.add_trace(t, row=(v//ncols)+1, col=(v % ncols)+1)

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
        height=nrows*400,
    ).update_traces(showlegend=False)

    return fig


def make_distplot_subplot(df, ncols=2):
    nrows = ceil((len(df.columns)) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[column for column in df.columns],
        vertical_spacing=0.03,
    )

    for v in range(len(df.columns)):
        values = df[df.columns[v]]
        label = [df.columns[v]]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        bin_width = (2 * iqr) / (len(values) ** (1 / 3))

        for t in ff.create_distplot([values], label, bin_size=bin_width).data:
            fig.add_trace(t, row=(v//ncols)+1, col=(v % ncols)+1)

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
        height=nrows*400,
    ).update_traces(showlegend=False)

    return fig


def make_scatter_target_subplots(df, ncols=2):
    nrows = ceil((len(df.columns)) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f'{column} vs Salary' for column in df.columns[:-1]],
        vertical_spacing=0.05,
        horizontal_spacing=0.08
    )

    for v in range(len(df.columns)-1):
        colum = df.columns[v]
        index = "" if v == 0 else f'{v+1}'
        fig['layout'][f'xaxis{index}']['title'] = colum
        fig['layout'][f'yaxis{index}']['title'] = "Salary"
        for t in px.scatter(df, x=df[colum], y="Salary").data:
            fig.add_trace(t, row=(v//ncols)+1, col=(v % ncols)+1)

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
        height=nrows*400,
    ).update_traces(showlegend=False)

    return fig
