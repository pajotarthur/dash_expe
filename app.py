# -*- coding: utf-8 -*-
import base64
from glob import glob

import dash
import dash_table_experiments as dt
from dash.dependencies import Input, Output

from db_tools import *
from utils import *

client = pymongo.MongoClient('drunk:27017')
app = dash.Dash()

external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",  # Normalize the CSS
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto"  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://cdn.rawgit.com/xhlulu/0acba79000a3fd1e6f552ed82edb8a64/raw/dash_template.css",
    "https://rawgit.com/plotly/dash-live-model-training/master/custom_styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

dbl = client.list_database_names()
database_list = []
for i in dbl:
    database_list.append({'label': i, 'value': i})

app.layout = html.Div([
    html.H1("Sacred Database",
            style={'display': 'inline',
                   'float': 'left',
                   'font-size': '2.65em',
                   'margin-left': '7px',
                   'font-weight': 'bolder',
                   'font-family': 'Product Sans',
                   'color': "rgba(117, 117, 117, 0.95)",
                   'margin-top': '20px',
                   'margin-bottom': '0'
                   }
            ),

    dcc.Dropdown(
        id='database',
        options=database_list,
        value='arthur_exp_database',
        searchable=False,
        clearable=False,
    ),

    html.Div(id='output-container'),

    dcc.Dropdown(
        id='expe',
        placeholder="select a expe",
        multi=True,
        clearable=True,
    ),

    dcc.Checklist(
        id='checklist',
        options=[
            {'label': 'Running', 'value': 'RUNNING'},
            {'label': 'Completed', 'value': 'COMPLETED'},
            {'label': 'Died', 'value': 'TIMEOUT'},
            {'label': 'Failed', 'value': 'FAILED'},
            {'label': 'Discriminator too small', 'value': 'DIS_TOO_SMALL'},
            {'label': 'Start Result too small', "value": "START_RESULT_TOO_SMALL"},
        ],
        values=['COMPLETED', 'RUNNING']
    ),
    html.Label('Max result'),
    html.Div(children=[
        html.Div(children=[
            dcc.Slider(
                id='range',
                min=0,
                max=5,
                step=0.01,
                value=1,
                marks={
                    0: {'label': '0', 'style': {'color': '#77b0b1'}},
                    0.05: {'label': '0.05', 'style': {'color': '#77b0b1'}},
                    0.1: {'label': '0.1', 'style': {'color': '#77b0b1'}},
                    0.25: {'label': '0.25'},
                    0.5: {'label': '0.5'},
                    1: {'label': '1'},
                    5: {'label': 'inf'},

                },

            )], style={'width': '75%', 'float': 'left', 'display': 'inline-block'}),
        html.Div(id='range_value', style={'width': '15%', 'float': 'right', 'display': 'inline-block'})
    ]),

    html.Div(
        children=[
            html.H4('Datatable '),
            dt.DataTable(
                # Initialise the rows
                rows=[{}],
                sortable=True,
                selected_row_indices=[],
                id='table')
        ], style={'marginBottom': 5, 'marginTop': 75}
    ),

    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Curve', children=[
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='expe_list_curve',
                        placeholder="select experiments",
                        multi=False,
                        clearable=True,
                    ),
                     dcc.Dropdown(
                        id='metric_list_curve',
                        placeholder="select metrics",
                        multi=False,
                        value='loss_MSE',
                        clearable=True,
                    ),
                ]),
        html.Div(id='run-log-storage', style={'display': 'none'}),
        div_graph('jojo')
        ])
        ]),
        dcc.Tab(label='Images', children=[
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        id='expe_list_image',
                        placeholder="select experiments",
                        multi=False,
                        clearable=True,
                    ),
                ]),
                dcc.RadioItems(id='train_or_test',
                    options=[
                        {'label':'Train', 'value': 'train'},
                        {'label':'Test', 'value': 'test'}
                    ], value='test'
                ),
                html.Div(children=[
                    html.Div(children=[
                        dcc.Slider(
                            id='slider_img',
                            min=0,
                            max=100,
                            step=1,
                            value=1,
                            marks={
                                0: {'label': '0', 'style': {'color': '#77b0b1'}},
                                25: {'label': '25', 'style': {'color': '#77b0b1'}},
                                50: {'label': '50'},
                                75: {'label': '75'},
                                100: {'label': '100'},
                            },

                        )], style={'width': '75%', 'float': 'left', 'display': 'inline-block'}),
                    html.Div(id='img_number', style={'width': '15%', 'float': 'right', 'display': 'inline-block'})
                ]),
                html.Img(id="image_exp")

            ])
        ]),
        dcc.Tab(label='Hyperparameters', children=[
            html.Div([
                dcc.RadioItems(
                    id='float_or_box',
                    options=[{'label': i, 'value': i} for i in ['scatter', 'box']],
                    value='scatter'
                ),
                dcc.Dropdown(
                    id='config',
                    placeholder="select an hyperparameter",
                    # multi=True,
                    clearable=True,
                ),
                html.Div(id='box-plot')
            ])
        ]),

    ], )

], style={'marginBottom': 250, 'marginTop': 25, 'marginLeft': 15, 'marginRight': 15})


@app.callback(
    Output('output-container', 'children'),
    [Input('database', 'value')])
def update_output(value):
    db = client[value]
    db_stat = db.command("dbstats")
    retour = "{} num_object : {} avgObjSize : {} datasize : {}".format(
        db_stat['db'],
        db_stat['objects'],
        convert_size(db_stat['avgObjSize']),
        convert_size(db_stat['dataSize']),
    )
    return retour

@app.callback(
    Output('expe', 'options'),
    [Input('database', 'value')])
def update_scrolldown(value):
    db = client[value]
    l = db.runs.distinct('experiment.name')
    c = []
    for i in l:
        c.append({'label': i, 'value': i})
    return c

@app.callback(
    Output('range_value', 'children'),
    [Input('range', 'value')])
def range_value(value):
    return "max : " + str(value)


@app.callback(Output('table', 'rows'),
              [Input('expe', 'value'),
               Input('database', 'value'),
               Input('checklist', 'values'),
               Input('range', 'value')])
def update_table(expe_name, value, completed, range_res):
    """
    For user selections, return the relevant table
    """
    db = client[value]
    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    if range_res < 5:
        filtre['result'] = {'$lt': range_res}

    if expe_name is not None:
        if db.runs.find(filtre).count() == 0:
            df = pd.DataFrame({'_id': [00], 'result': [10000], 'start_time': ['2018-07-02 09:58:15.077000'], 'status': ['FAILED'], 'experiment.name': ['NO EXPERIMENT'], 'host.hostname': ['NONE']})
            return df.to_dict('records')

        df = get_results(db.runs, project={'start_time': True,
                                           "status": True,
                                           "host.hostname": True,
                                           "experiment.name": True
                                           },
                         filter_by=filtre, include_index=True, prune=False)
    else:
        df = pd.DataFrame({'A' : []})

    return df.to_dict('records')

#
# TAB HYPERPARAMATERS
@app.callback(Output('config', 'options'),
              [
                  Input('expe', 'value'),
                  Input('float_or_box', 'value'),
                  Input('database', 'value'),
                  Input('checklist', 'values'),
                  Input('range', 'value'),
              ])
def update_config_name(expe_name, float_or_box, value, completed, range_res):
    db = client[value]
    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    filtre['result'] = {'$lt': range_res}
    l_hyper = []
    skip_cols = ["config.device", "config.seed", "config.niter", "result"]
    list_box_not_scatter = ["config.gen.nz", "config.optim_dis.lr", "config.optim_gen.lr", "config.closure.nz"]
    if expe_name is not None:
        if db.runs.find(filtre).count() > 0:
            df = get_results(db.runs, filter_by=filtre, include_index=True)

            for i in df.columns:

                if i in skip_cols:
                    continue
                print(i)
                if (df[i].dtype == np.bool or df[i].dtype == np.object_ or i in list_box_not_scatter) and float_or_box == 'box':
                    val = i[7:]
                    l_hyper.append({'label': val, 'value': i}, )
                if (df[i].dtype == np.float or df[i].dtype == np.int) and float_or_box == 'scatter':
                    val = i[7:]
                    l_hyper.append({'label': val, 'value': i}, )

    if len(l_hyper) == 0:
        if float_or_box == 'box':
            l_hyper.append({'label': 'pas de box', value: 0})
        if float_or_box == 'scatter':
            l_hyper.append({'label': 'pas de scatter', value: 0})
    return l_hyper


@app.callback(Output('box-plot', 'children'),
              [
                  Input('config', 'value'),
                  Input('float_or_box', 'value'),
                  Input('expe', 'value'),
                  Input('database', 'value'),
                  Input('checklist', 'values'),
                  Input('range', 'value')
              ])
def update_config_plot(box_value, float_or_box, expe_name, value, completed, range_res):
    db = client[value]

    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    filtre['result'] = {'$lt': range_res}
    if expe_name is None:
        return
    if db.runs.find(filtre).count() == 0:
      return
    df = get_results(db.runs, filter_by=filtre, include_index=True)
    data = []
    if box_value is None:
        return "Il n'y a rien a afficher pour l'instant"

    if float_or_box == 'scatter':
        fig = float_to_scatter(df, box_value)
    else:
        df[box_value] = df[box_value].fillna(value='nan')
        df[box_value] = pd.Categorical(df[box_value])
        data += cat_to_boxplot(df, box_value)

        layout = go.Layout(
            title="Hyperparameter"
        )
        fig = go.Figure(data=data, layout=layout)

    g = dcc.Graph(figure=fig, id='coco_lasticot')

    return g


# # TAB CURVE
@app.callback(Output('expe_list_curve', 'options'),
              [Input('expe', 'value'),
               Input('database', 'value'),
               Input('checklist', 'values'),
               Input("range", 'value')])
def update_expe_list_curve(expe_name, value, completed, range_res):
    db = client[value]

    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    if range_res < 5:
        filtre['result'] = {'$lt': range_res}
    l_retour = []

    if expe_name is not None:
        if db.runs.find(filtre).count() > 0:

            df = get_results(db.runs, filter_by=filtre, project={'start_time': True,
                                                                 "status": True,
                                                                 "host.hostname": True,
                                                                 "experiment.name": True
                                                                 }, include_index=True, prune=False)
            for row in df.iterrows():
                l_retour.append({'label': "{}_{}_{}".format(row[1]["_id"],
                                                            row[1]["result"],
                                                            row[1]["experiment.name"]),
                                 'value': row[1]["_id"]})


    return sorted(l_retour, key=lambda k: k['label'].split('_')[1])



@app.callback(Output('metric_list_curve', 'options'),
              [
               Input('expe_list_curve', 'value'),
               Input('expe', 'value'),
               Input('database', 'value'),
               Input('checklist', 'values'),
               Input('range', 'value')])
def update_metrics_list_curve(expe_id, expe_name, value, completed, range_res):
    db = client[value]
    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    filtre['_id'] = expe_id
    list_metric_name = []

    if expe_name is not None:
        if db.runs.find(filtre).count() > 0:

            for l in db.runs.find(filtre):
                metrics = l['info']['metrics']

            for m in metrics:
              list_metric_name.append(m['name'].split('/')[1])
            list_metric_name = np.unique(list_metric_name)
        else:
            list_metric_name = ['no metrics']

    return [{'label': i, 'value':i} for i in list_metric_name]


@app.callback(Output('run-log-storage', 'children'),
              [
               Input('expe_list_curve', 'value'),
               Input('expe', 'value'),
               Input('database', 'value'),
               Input('checklist', 'values'),
               Input('range', 'value'),])
def get_run_log(expe_id, expe_name, value, completed, range_res):
    db = client[value]
    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    filtre['result'] = {'$lt': range_res}
    filtre['_id'] = expe_id
    json = ''
    if expe_name is not None:
        if db.runs.find(filtre).count() == 0:
          return

        for l in db.runs.find(filtre):
            metrics = l['info']['metrics']


        df_dict = {}
        for i in metrics:
            n = i['name']
            for kk in db.metrics.find({'_id': ObjectId(i['id'])}):
                v = kk['values']
            df_dict[n] = v
            df_dict['step'] = kk['steps']
        run_log_df = pd.DataFrame(df_dict)
        try:
            json = run_log_df.to_json(orient='split')
        except FileNotFoundError as error:
            print(error)
            print("Please verify if the csv file generated by your model is placed in the correct directory.")
            return None

    return json


@app.callback(Output('div-jojo-graph', 'children'),
              [Input('run-log-storage', 'children'),
               Input('radio-display-mode-jojo', 'value'),
               Input('checklist-smoothing-options-jojo', 'values'),
               Input('slider-smoothing-jojo', 'value'),
               Input('metric_list_curve', 'value')])
def update_accuracy_graph(log_storage, display_mode,
                          checklist_smoothing_options,
                          slider_smoothing,
                          metric_name):

    graph = update_graph('accuracy-graph',
                         metric_name,
                         'meters/'+metric_name+'/train',
                         'meters/'+metric_name+'/test',
                         log_storage,
                         display_mode,
                         checklist_smoothing_options,
                         slider_smoothing,
                         metric_name)

    try:
        if display_mode in ['separate_horizontal', 'overlap']:
            graph.figure.layout.yaxis['range'] = [0, 1]
        else:
            graph.figure.layout.yaxis1['range'] = [0, 1]
            graph.figure.layout.yaxis2['range'] = [0, 1]

    except AttributeError:
        pass

    return [graph]
#
# # IMAGE TAB
#
#
@app.callback(
    Output('img_number', 'children'),
    [Input('slider_img', 'value')])
def range_value(value):
    return "img : " + str(value)


@app.callback(Output('expe_list_image', 'options'),
              [
                  Input('expe', 'value'),
                  Input('database', 'value'),
                  Input('checklist', 'values'),
                  Input('range', 'value')
              ])
def update_expe_list_image(expe_name, value, completed, range_res):
    db = client[value]

    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    if range_res < 5:
        filtre['result'] = {'$lt': range_res}
    l_retour = []

    if expe_name is not None:
        if db.runs.find(filtre).count() > 0:

            df = get_results(db.runs, filter_by=filtre, project={'start_time': True,
                                                                 "status": True,
                                                                 "host.hostname": True,
                                                                 "experiment.name": True
                                                                 }, include_index=True, prune=False)
            for row in df.iterrows():
                l_retour.append({'label': "{}_{}_{}".format(row[1]["_id"],
                                                            row[1]["result"],
                                                            row[1]["experiment.name"]),
                                 'value': row[1]["_id"]})


    return sorted(l_retour, key=lambda k: k['label'].split('_')[1])



@app.callback(Output('slider_img', 'max'),
              [
                  Input('expe', 'value'),
                  Input('database', 'value'),
                  Input('checklist', 'values'),
                  Input('range', 'value'),
                  Input('expe_list_image', 'value'),
                  Input('train_or_test', 'value')
              ])
def update_image_slider(expe_name, value, completed, range_res, id, train_or_test):
    if id is None:
      return

    db = client[value]
    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    if range_res < 5:
        filtre['result'] = {'$lt': range_res}
    length = 0
    if expe_name is not None:
        if db.runs.find(filtre).count() > 0:
            df = get_results(db.runs, filter_by=filtre, project={'start_time': True,
                                                                 "status": True,
                                                                 "host.hostname": True,
                                                                 "experiment.name": True,
                                                                 "info.exp_dir": True,
                                                                 }, include_index=True,
                             prune=False).sort_values('result')

            folder = df[df['_id'] == id]['info.exp_dir'].values[0]
            if folder is None:
                return 0
            if train_or_test == 'test':
                length = len(glob(folder + "/test*"))
                if length == 0:
                  folder = folder.replace('big', 'gogos')
                  length = len(glob(folder + "/test*"))

            elif train_or_test == 'train':
                length = len(glob(folder + "/train*"))
                if length == 0:
                  folder = folder.replace('big', 'gogos')
                  length = len(glob(folder + "/train*"))

            else:
                length = 0

    return length


@app.callback(Output('image_exp', 'src'),
              [
                  Input('expe', 'value'),
                  Input('database', 'value'),
                  Input('checklist', 'values'),
                  Input('range', 'value'),
                  Input('expe_list_image', 'value'),
                  Input('slider_img', 'value'),
                  Input('train_or_test', 'value')
              ])
def update_image(expe_name, value, completed, range_res, id, slider_num, train_or_test):

    if id is None:
      return

    db = client[value]

    filtre = {'experiment.name': {'$in': expe_name}}
    filtre['status'] = {'$in': completed}
    if range_res < 5:
        filtre['result'] = {'$lt': range_res}
    if expe_name is not None:
        if db.runs.find(filtre).count() > 0:
            df = get_results(db.runs, filter_by=filtre, project={'start_time': True,
                                                                 "status": True,
                                                                 "host.hostname": True,
                                                                 "experiment.name": True,
                                                                 "info.exp_dir": True,
                                                                 },
                             include_index=True,
                             prune=False).sort_values('result')

            folder = df[df['_id'] == id]['info.exp_dir'].values[0]
            if folder is None:
                return []
            if train_or_test == 'test':
                length = len(glob(folder + "/test*"))
                list_img = glob(folder+"/test*")
                if length == 0:
                  folder = folder.replace('big', 'gogos')
                  length = len(glob(folder + "/test*"))
                  list_img = glob(folder+"/test*")


            elif train_or_test == 'train':
                length = len(glob(folder + "/train*"))
                list_img = glob(folder+"/train*")
                if length == 0:
                  folder = folder.replace('big', 'gogos')
                  length = len(glob(folder + "/train*"))
                  list_img = glob(folder+"/train*")
            else:
                list_img = []

            if len(list_img) > 0:
                list_img = sorted_nicely(list_img)
                encoded_image = base64.b64encode(open(list_img[slider_num], 'rb').read()).decode('utf-8').replace('\n', '')
                return 'data:image/png;base64,{}'.format(encoded_image)
            else:
                return ''
        else:
            return ''
    else:
        return ''


if __name__ == '__main__':
    app.run_server(debug=True)
