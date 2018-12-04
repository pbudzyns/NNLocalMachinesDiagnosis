import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import plotly.graph_objs as go
from collections import deque
import pandas as pd
import time
import os
import agate
import glob
from plotly.offline import iplot
import plotly.io as pio
import plotly


external_stylesheets = ['https://codepen.io/pawelbpw/pen/aRrJRj.css']
plotly.io.orca.config.executable = r"C:\Users\pbudzyns\AppData\Local\Programs\orca\orca.exe"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)






app.layout = html.Div([
    dcc.Graph(id='stat-graph', animate=True),
])

# @app.callback(Output('stat-graph', 'figure'),
#               inputs=[Input('my-dropdown', 'value')])
# def update_chart(filename):
#     figure = get_figure(filename)
#     return figure

def get_figure(filename, type='dashboard'):
    data = []
    table = pd.read_csv(filename)
    # tmp_table = pd.read_csv("outputs/snr_to_acc(2,).csv")
    # x = list(table['SNR'])
    # infos = "ACC,RECALL,PRECISION,F1_SCORE".split(',')
    # for info in infos:
    #     d = go.Scatter(x=x, y=list(table[info]),
    #                    name=info, mode='lines+markers')
    #     data.append(d)
    if type=="box":
        for index, row in table.iterrows():
            tmp = row["Proba"].replace("[", "").replace("]", "").split(",")
            t = [float(t) for t in tmp]
            # print(t)
            d = go.Box(name=row["IMP"], y=t, marker=dict(color='rgb(8, 81, 156)',))
            data.append(d)
        layout = go.Layout(xaxis=dict(title='Srednia wysokosc impulsu'), yaxis=dict(range=(0, 1.1), title='Predykowane prawdopodobienstwo uszkodzenia'),
                           showlegend=False)
        return {'data': data, 'layout': layout}
    if type=="bar":
        x = list(table["IMP"])
        x[0] = 0
        data = go.Bar(x=x, y=list(table["Hits"]), marker=dict(color='rgb(8, 81, 156)',))
        layout = go.Layout(xaxis=dict(title='Srednia wysokosc impulsu'), yaxis=dict(range=(0, 100), title='% predykcji jako uszkodzony'))
        return {'data': [data], 'layout': layout}

def plot_all(filenames, type):
    for filename in filenames:
        save_as_pdf(filename, type)

def save_as_pdf(filename, type=None):
    figure = get_figure(filename, type)
    new_filename = filename.replace("outputs", "../../latex/images2").replace(".csv", f"{type}.pdf")
    pio.write_image(figure, new_filename)




if __name__ == '__main__':
    # plot_all(glob.glob("outputs/nn*.csv"))
    # app.run_server(port=8080, debug=True)
    save_as_pdf("inputs/monitor_system_stats300_with_probas2.csv", type="bar")
