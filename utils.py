import math

import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from plotly import tools

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])


def div_graph(name):
    """Generates an html Div containing graph and control options for smoothing and display, given the name"""
    return html.Div([
        html.Div(
            id=f'div-{name}-graph',
            className="ten columns"
        ),

        html.Div([
            html.Div([
                html.P("Smoothing:", style={'font-weight': 'bold', 'margin-bottom': '0px'}),

                dcc.Checklist(
                    options=[
                        {'label': ' Training', 'value': 'train'},
                        {'label': ' Validation', 'value': 'val'}
                    ],
                    values=[],
                    id=f'checklist-smoothing-options-{name}'
                )
            ],
                style={'margin-top': '10px'}
            ),

            html.Div([
                dcc.Slider(
                    min=0,
                    max=1,
                    step=0.05,
                    marks={i / 5: i / 5 for i in range(0, 6)},
                    value=0.6,
                    updatemode='drag',
                    id=f'slider-smoothing-{name}'
                )
            ],
                style={'margin-bottom': '40px'}
            ),

            html.Div([
                html.P("Plot Display mode:", style={'font-weight': 'bold', 'margin-bottom': '0px'}),

                dcc.RadioItems(
                    options=[
                        {'label': ' Overlapping', 'value': 'overlap'},
                        {'label': ' Separate (Vertical)', 'value': 'separate_vertical'},
                        {'label': ' Separate (Horizontal)', 'value': 'separate_horizontal'}
                    ],
                    value='overlap',
                    id=f'radio-display-mode-{name}'
                ),

                html.Div(id=f'div-current-{name}-value')
            ]),
        ],
            className="two columns"
        ),
    ],
        className="row"
    )


def update_graph(graph_id,
                 graph_title,
                 y_train_index,
                 y_val_index,
                 run_log_json,
                 display_mode,
                 checklist_smoothing_options,
                 slider_smoothing,
                 yaxis_title):
    """
    :param graph_id: ID for Dash callbacks
    :param graph_title: Displayed on layout
    :param y_train_index: name of column index for y train we want to retrieve
    :param y_val_index: name of column index for y val we want to retrieve
    :param run_log_json: the json file containing the data
    :param display_mode: 'separate' or 'overlap'
    :param checklist_smoothing_options: 'train' or 'val'
    :param slider_smoothing: value between 0 and 1, at interval of 0.05
    :return: dcc Graph object containing the updated figures
    """

    def smooth(scalars, weight=0.6):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    if run_log_json:  # exists
        layout = go.Layout(
            title=graph_title,
            margin=go.layout.Margin(l=50, r=50, b=50, t=50),
            yaxis={'title': yaxis_title, 'range': [0.05, 0.4]}
        )

        run_log_df = pd.read_json(run_log_json, orient='split')
        step = run_log_df['step']
        y_train = run_log_df[y_train_index]
        y_val = run_log_df[y_val_index]

        # Apply Smoothing if needed
        if 'train' in checklist_smoothing_options:
            y_train = smooth(y_train, weight=slider_smoothing)

        if 'val' in checklist_smoothing_options:
            y_val = smooth(y_val, weight=slider_smoothing)

        trace_train = go.Scatter(
            x=step,
            y=y_train,
            mode='lines',
            name='Training'
        )

        trace_val = go.Scatter(
            x=step,
            y=y_val,
            mode='lines',
            name='Validation'
        )

        if display_mode == 'separate_vertical':
            figure = tools.make_subplots(rows=2,
                                         cols=1,
                                         print_grid=False,
                                         shared_yaxes=True)

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 2, 1)

            figure['layout'].update(title=layout.title,
                                    margin=layout.margin,
                                    scene={'domain': {'x': (0., 0.5), 'y': (0.5,1)}})

        elif display_mode == 'separate_horizontal':
            figure = tools.make_subplots(rows=1,
                                         cols=2,
                                         shared_yaxes=True,
                                         print_grid=False)

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 1, 2)

            figure['layout'].update(title=layout.title,
                                    margin=layout.margin)

        elif display_mode == 'overlap':
            figure = go.Figure(
                data=[trace_train, trace_val],
                layout=layout
            )

        else:
            figure = None

        return dcc.Graph(figure=figure, id=graph_id)

    return dcc.Graph(id=graph_id)