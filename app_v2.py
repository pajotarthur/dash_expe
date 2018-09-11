import copy

import dash
from dash.dependencies import Input, Output

from db_tools import *
from utils import *

client = pymongo.MongoClient('drunk:27017')
app = dash.Dash()
db = client.arthur_exp_database.runs
dbl = client.arthur_exp_database.runs.find().distinct("experiment.name")

database_list = []
for i in dbl:
    database_list.append({'label': i, 'value': i})

layout = dict(
    autosize=True,
    height=300,
    margin=dict(
        l=35,
        r=35,
        b=35,
        t=45
    ),
    hovermode="closest",
    legend=dict(font=dict(size=10), orientation='h'),
)


def serve_layout():
    return html.Div([
        # Banner display
        html.Div([
            html.H2(
                'Sacred Database Viz',
                id='title'
            ),
            html.Img(
                src="http://fetedelascience.lip6.fr/images/su_logo.png"
            )
        ],
            className="banner"
        ),
        html.Div(
            [
                html.H5(
                    '',
                    id='exp_count',
                    className='six columns'
                ),

                html.H5(
                    '',
                    id='res_range_text',
                    className='six columns',
                    style={'text-align': 'right'}
                ),
            ],
            className='row'
        ),
        html.Div(
            [
                dcc.Dropdown(
                    id='expe',
                    placeholder="select a expe",
                    multi=True,
                    clearable=False,
                    options=database_list,
                ),
            ],
            style={'margin-top': '20'}

        ),

        html.Div(
            [
                html.P('Filter by result range  (or select range in histogram):'),
                html.Div(
                    [   dcc.RangeSlider(
                            id='result_slider',
                            min=0,
                            max=2,
                            step=0.01,
                            value=[0,1],
                            marks={
                                0: {'label': '0', 'style': {'color': '#77b0b1'}},
                                0.05: {'label': '0.05', 'style': {'color': '#77b0b1'}},
                                0.1: {'label': '0.1', 'style': {'color': '#77b0b1'}},
                                0.25: {'label': '0.25'},
                                0.5: {'label': '0.5'},
                            },),
                        dcc.Graph(id='count_graph')
                    ],
                    style={'margin-top': '10'}
                ),
            ],
            style={'margin-top': '20'}
        ),

    ]
    className='ten columns offset-by-one'

    )


app.layout = serve_layout()


# Slider -> result text
@app.callback(Output('res_range_text', 'children'),
              [Input('result_slider', 'value')])
def update_range_text(range_slider):
    return "{} | {} ".format(range_slider[0], range_slider[1])

# Selectors -> count graph
@app.callback(Output('count_graph', 'figure'),
              [
                  Input('expe', 'value'),
                  Input('result_slider', 'value')
              ])
def make_count_figure(expe, slider):

    filtre = {'experiment.name': {'$in': expe}, 'result':{'$lt': 2}}

    l = []
    for i in db.find(filtre):
        l.append(i['result'])

    colors = []
    for i in np.arange(0, 2, 0.01):

        if i >= int(slider[0]) and i < int(slider[1]):
            colors.append('rgb(192, 0, 27)')
        else:
            colors.append('rgba(192, 0, 27, 0.2)')

    g = []
    for i in np.arange(0, 2, 0.01):
        l_i = []
        for j in l:
            if j >= i and j < i + 0.01:
                l_i.append(j)
        g.append(len(l_i))

    g = np.array(g)

    data = [
        dict(
            type='scatter',
            mode='markers',
            x=np.arange(0, 2, 0.01),
            y=g * 2,
            name='All Wells',
            opacity=0,
            hoverinfo='skip'
        ),
        dict(
            type='bar',
            x=np.arange(0, 2, 0.01),
            y=g,
            name='All Wells',
            marker=dict(
                color=colors
            ),
        ),
    ]
    layout_count = copy.deepcopy(layout)
    layout_count['title'] = 'Completed Wells/Year'
    layout_count['dragmode'] = 'select'
    layout_count['showlegend'] = False
    #
    figure = dict(data=data, layout=layout_count)
    return figure


# ######################################### CSS #########################################
# external_css = [
#     "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",  # Normalize the CSS
#     "https://fonts.googleapis.com/css?family=Open+Sans|Roboto"  # Fonts
#     "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
#     "https://cdn.rawgit.com/TahiriNadia/styles/faf8c1c3/stylesheet.css",
#     "https://cdn.rawgit.com/TahiriNadia/styles/b1026938/custum-styles_phyloapp.css"
# ]
#
# for css in external_css:
#     app.css.append_css({"external_url": css})
# app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501


# Running the server
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)