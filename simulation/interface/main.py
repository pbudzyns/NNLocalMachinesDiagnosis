import dash
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

from simulation.signal_source.signal_source import SignalSource

# X = deque(maxlen=20)
# Y = deque(maxlen=20)
# X.append(1)
# Y.append(1)

app = dash.Dash(__name__)
app.layout = html.Div([dcc.Graph(id='live-graph', animate=True),
                       dcc.Interval(id='graph-update', interval=1000)])



@app.callback(Output('live-graph', 'figure'),
              events=[Event('graph-update', 'interval')])
def update_graph():

    # X.append(X[-1]+1)
    # Y.append(Y[-1]+Y[-1]*random.uniform(-0.1, 0.1))
    signal, t = signal_source.get_signal(0.75)

    data = go.Scatter(x=t, y=signal, name='Scatter',
                      mode='lines')

    return {'data': [data], 'layout': go.Layout(xaxis=dict(range=[min(t), max(t)]),
                                                yaxis=dict(range=[-5, 5]))}

if __name__ == '__main__':
    signal_source = SignalSource(imp_amp=4, interval=1.5)
    try:
        signal_source.start_thread()
        app.run_server(debug=True)
    except:
        pass

