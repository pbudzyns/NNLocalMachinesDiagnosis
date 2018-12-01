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

file_list = [{'label':name.replace("outputs/", ""), 'value':name} for name in glob.glob("outputs/nn*.csv")]




app.layout = html.Div([
    dcc.Dropdown(
        id='my-dropdown',
        options=file_list,
        value=file_list[0]['label'],
        clearable=False,
    ),
    dcc.Graph(id='stat-graph', animate=True),
])

@app.callback(Output('stat-graph', 'figure'),
              inputs=[Input('my-dropdown', 'value')])
def update_chart(filename):
    figure = get_figure(filename)
    return figure

def get_figure(filename, type='dashboard'):
    data = []
    table = pd.read_csv(filename)
    tmp_table = pd.read_csv("outputs/snr_to_acc(2,).csv")
    x = list(tmp_table['SNR'])
    infos = "ACC,RECALL,PRECISION,F1_SCORE".split(',')
    for info in infos:
        d = go.Scatter(x=x, y=list(table[info]),
                       name=info, mode='lines+markers')
        data.append(d)

    if type == 'dashboard':
        layout = go.Layout(xaxis=dict(title='SNR'), yaxis=dict(range=(0, 1.1), title='Value'))
    else:
        layout = go.Layout(xaxis=dict(title='SNR'), yaxis=dict(range=(0, 1.1), title='Value'), autosize=False,
                           width=1000, height=500)

    return {'data': data, 'layout': layout}

def plot_all(filenames):
    for filename in filenames:
        save_as_pdf(filename)

def save_as_pdf(filename):
    figure = get_figure(filename, type='pdf')
    new_filename = filename.replace("outputs", "../../latex/images2").replace(".csv", ".pdf")
    pio.write_image(figure, new_filename)




if __name__ == '__main__':
    plot_all(glob.glob("outputs/nn*.csv"))
    app.run_server(port=8080, debug=True)
