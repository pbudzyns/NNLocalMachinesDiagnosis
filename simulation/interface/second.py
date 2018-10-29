import dash
from dash.dependencies import Output, Event, Input, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from collections import deque

from simulation.signal_source.signal_source import SignalSource
from simulation.analytics.monitor import Monitor

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
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

app.layout = html.Div([
    html.Div(
        className="row",
        children=[
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
                    html.H6('Pulsation'),
                    # html.Br(),
                    dcc.Input(id='pulsation-source', placeholder='Enter a value ...', value='4.0', type='text'),
                    html.Button('Refresh', id='refresh-button')
                ], ),
                style={'width': '20%'}
            )
        ]
    )
])

                       # dcc.Interval(id='graph-update', interval=1000),
                       # dcc.Slider(id='imp_slider', min=0, max=5, step=0.1, value=0),
                       # html.Div(id='slider-output-container'),
                       # dcc.Graph(id='live-prediction-graph', animate=True),
                       # dcc.Interval(id='prediction-graph-update', interval=1000)


monitor = Monitor()
monitor.load_model("../analytics/models/mlp_classifier_one_big.model")
signal_source = SignalSource()
h_proba = deque(maxlen=20)
d_proba = deque(maxlen=20)
n = deque(maxlen=20)
n.append(0)
# signal = [0, ]
# t = [0, ]

# @app.callback(Output('live-graph', 'figure'), inputs=[Input('refresh-button', 'n_clicks')],
#               state=[State('pulsation-source', 'value')])
def plot_signal(n_clicks, pulsation):
    try:
        pulsation = float(pulsation)
    except ValueError:
        print("Couldn't get float from string")
        pulsation = 0.0
    # print('Signal prepared')
    # print('signal: \n', signal, '\n T: \n', t)
    signal, t = signal_source.get_single_signal(pulsation=pulsation, duration=2)
    data = go.Scatter(x=list(t), y=list(signal), name='Signal', mode='lines')
    layout = go.Layout(xaxis=dict(range=(min(t), max(t))), yaxis=dict(range=(-5, 5)))

    return {'data': [data], 'layout': layout}
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

if __name__ == '__main__':

    # signal_source.start_thread()

    app.run_server(port=8080, debug=True)
