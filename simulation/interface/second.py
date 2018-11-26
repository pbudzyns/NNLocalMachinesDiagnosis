import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
from dash_table_experiments import DataTable
import plotly.graph_objs as go
from collections import deque
import pandas as pd
import time
import os

from simulation.signal_source.signal_source import SignalSource
from simulation.analytics.monitor import Monitor

hover_table = """\ntr:hover {background-color:#f5f5f5;}\n"""

external_stylesheets = ['https://codepen.io/pawelbpw/pen/aRrJRj.css']


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# abs_path = os.path.abspath(os.curdir)
# print(abs_path)

app.scripts.config.serve_locally = True
# app.layout = html.Div([
#               # html.H1('Signal Visualisation'),
#
#                 html.Div([
#                           dcc.Graph(id='live-graph', animate=True),
#                          ], style={'width': '60%', 'display': 'inline-block'}),
#                 html.Div([html.Div([
#                        dcc.Input(id='pulsation-source', placeholder='Enter a value ...', value='4.0', type='text'),
#                        html.Button('Refresh', id='refresh-button'),
#                        # html.Div(id='message-container')
#                          ])], style={'width': '35%', 'display': 'inline-block', 'vertical-align': 'middle'}),
#                       ])

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ],) for i in range(min(len(dataframe), max_rows))],

    )

result_table = pd.DataFrame(columns = ['ID', 'Pulsacja', 'Czas trwania', 'Prawdopodobieństwo uszkodzenia', 'Klasyfikacja'])
# result_table.loc[0] = [0,0,0,0]
app.layout = html.Div([
    html.Div(
        className="row",
        children=[
            html.H1('System Monitorujący'),
            html.Div(
                className="six columns",
                children=[
                    html.Div(
                        children=dcc.Graph(id='live-graph', animate=True),

                    )
                ],
                style={'width': '70%'}
            ),
            html.Div(
                className="six columns",
                children=html.Div([
                    html.H6('Pulsacje'),
                    dcc.Input(id='pulsation-source', placeholder='Enter a value ...', value='2.5', type='text'),
                    html.H6('Długość sygnału'),
                    dcc.Input(id='duration-source', placeholder='Enter a value ...', value='0.5', type='text'),
                    html.Br(),
                    html.Button('Generate', id='refresh-button'),
                ],
                style={'margin': 'auto', 'position': 'absolute', 'top': '20%'}
                ),
                style={'width': '20%'}
            ),
        ]
    ),
    html.Div(children=[
                html.Div(id='table-handler', children=None, style={'margin': 'auto'} )
        ],
        style={'margin': 'auto', 'width': '50%'}
    ),
])

monitor = Monitor()
monitor.load_model("../analytics/models/mlp_classifier_one_big_5.model")
signal_source = SignalSource()
h_proba = deque(maxlen=20)
d_proba = deque(maxlen=20)
n = deque(maxlen=20)
n.append(0)
signal = [0,]
# t = [0, ]

@app.callback(Output('live-graph', 'figure'),
              inputs=[Input('refresh-button', 'n_clicks'), Input('pulsation-source', 'n_submit'),
                      Input('duration-source', 'n_submit')],
              state=[State('pulsation-source', 'value'), State('duration-source', 'value')])
def plot_signal(n_clicks, n_submit, n_submit2, pulsation, duration):
    global signal
    try:
        pulsation = float(pulsation.replace(',', '.'))
        duration = float(duration.replace(',', '.'))
    except ValueError:
        print("Couldn't get float from string")
        pulsation = 0.0
        duration = 2.0
    # print('Signal prepared')
    # print('signal: \n', signal, '\n T: \n', t)
    signal, t = signal_source.get_single_signal(pulsation=pulsation, duration=duration)
    data = go.Scatter(x=list(t), y=list(signal), name='Sygnał', mode='lines')
    layout = go.Layout(xaxis=dict(range=(min(t), max(t)), title='Time'), yaxis=dict(range=(-5, 5), title='Amplituda'),
                        title='Sygnał diagnostyczny')

    return {'data': [data], 'layout': layout}

@app.callback(Output('table-handler', 'children'),
              inputs=[Input('refresh-button', 'n_clicks'), Input('pulsation-source', 'n_submit'),
                      Input('duration-source', 'n_submit')],
              state=[State('pulsation-source', 'value'), State('duration-source', 'value')])
def update_table(n_clicks, n_submit, n_submit2, pulsation, duration):
    global signal, result_table
    time.sleep(0.1)
    h_proba, d_proba = monitor.get_damage_proba(signal)
    status = 'Uszkodzony' if d_proba > h_proba else 'Sprawny'
    result_table.loc[-1] = [0, pulsation, duration, '%.3f'%(d_proba), status]  # adding a row
    result_table.index = result_table.index + 1
    result_table['ID'] += 1
    result_table = result_table.sort_index()
    return [generate_table(result_table)]
#
# @app.callback(Output('slider-output-container', 'children'),
#               inputs=[Input('imp_slider', 'value')])
# def update_output(value):
#     signal_source.set_pulsation(float(value))
#     return f"Current pulsation: {value}"
#
#
# @app.callback(Output('live-prediction-graph', 'figure'),
#               events=[Event('prediction-graph-update', 'interval')])
# def update_prediction_graph():
#     global signal, t
#     p1, p2 = monitor.get_damage_proba(signal)
#     h_proba.append(p1)
#     d_proba.append(p2)
#     n.append(n[-1] + 1)
#
#     h_data = go.Scatter(x=list(n), y=list(h_proba), name='H_Scatter', mode='lines+markers')
#     d_data = go.Scatter(x=list(n), y=list(d_proba), name='D_Scatter', mode='lines+markers')
#
#     return {'data': [h_data, d_data], 'layout': go.Layout(xaxis=dict(range=[min(n), max(n)]),
#                                                           yaxis=dict(range=[0, 1]))}
#
#
# @app.callback(Output('live-graph', 'figure'),
#               events=[Event('graph-update', 'interval')])
# def update_graph():
#     global signal, t
#
#     signal, t = signal_source.get_signal(0.75)
#
#     data = go.Scatter(x=t, y=signal, name='Scatter',
#                       mode='lines')
#
#     return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(t), max(t)]),
#                                                 yaxis=dict(range=[-5, 5]))}
#

# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })

if __name__ == '__main__':

    # signal_source.start_thread()

    app.run_server(port=8080, debug=True)
