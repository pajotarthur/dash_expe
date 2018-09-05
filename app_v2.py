import dash
import base64
from glob import glob

from db_tools import *
from utils import *

from dash.dependencies import Input, Output, State


client = pymongo.MongoClient('drunk:27017')
app = dash.Dash()
dbl = client.arthur_exp_database.runs.find().distinct("experiment.name")

database_list = []
for i in dbl:
    database_list.append({'label': i, 'value': i})

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
                    [
                        dcc.Graph(id='count_graph')
                    ],
                    style={'margin-top': '10'}
                ),
            ],
            style={'margin-top': '20'}
        ),

    ],
    className='ten columns offset-by-one'

    )


app.layout = serve_layout()


# # Slider -> result text
# @app.callback(Output('res_range_text', 'children'),
#               [Input('result_slider', 'value')])
# def update_year_text(year_slider):
#     return " 0 | {} ".format(year_slider)

# Selectors -> count graph
@app.callback(Output('count_graph', 'figure'),
              [Input('expe', 'value')])
def make_count_figure(expe):

    print(expe)
    filtre = {'experiment.name': {'$in': expe}}
    # layout_count = copy.deepcopy(layout)
    #
    # dff = filter_dataframe(df, well_statuses, well_types, [1960, 2017])
    # g = dff[['API_WellNo', 'Date_Well_Completed']]
    # g.index = g['Date_Well_Completed']
    # g = g.resample('A').count()
    #
    # colors = []
    # for i in range(1960, 2018):
    #     if i >= int(year_slider[0]) and i < int(year_slider[1]):
    #         colors.append('rgb(192, 255, 245)')
    #     else:
    #         colors.append('rgba(192, 255, 245, 0.2)')
    #
    # data = [
    #     dict(
    #         type='scatter',
    #         mode='markers',
    #         x=g.index,
    #         y=g['API_WellNo'] / 2,
    #         name='All Wells',
    #         opacity=0,
    #         hoverinfo='skip'
    #     ),
    #     dict(
    #         type='bar',
    #         x=g.index,
    #         y=g['API_WellNo'],
    #         name='All Wells',
    #         marker=dict(
    #             color=colors
    #         ),
    #     ),
    # ]
    #
    # layout_count['title'] = 'Completed Wells/Year'
    # layout_count['dragmode'] = 'select'
    # layout_count['showlegend'] = False
    #
    # figure = dict(data=data, layout=layout_count)
    # return figure
    return None

######################################### CSS #########################################
external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",  # Normalize the CSS
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto"  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://cdn.rawgit.com/TahiriNadia/styles/faf8c1c3/stylesheet.css",
    "https://cdn.rawgit.com/TahiriNadia/styles/b1026938/custum-styles_phyloapp.css"

]

for css in external_css:
    app.css.append_css({"external_url": css})
app.css.append_css({'external_url': 'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.css'})  # noqa: E501


# Running the server
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)