# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from textwrap import dedent

from addict import Dict

import argparse
import yaml
import glob
import flask
import dash
import dash_daq as daq
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.figure_factory as ff

app = dash.Dash(__name__)
server = app.server # the Flask app
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True


# ============================ Variables ===============================

args_paths = glob.glob('{}/*/{}*/apply_eyeliner'.format('demo/results', 'lap_'))
arglist = [os.path.basename(os.path.dirname(x)) for x in args_paths]
laplist = [os.path.basename(x)[-2] for x in args_paths]
arglaplist = []
for arg, lap in zip(arglist, laplist):
    arglaplist.append(arg+"/lap_"+lap)
video_length = 400
VIDEOS_PATH = os.getcwd()+"/"
colorscale = [[0, '#0033FF'], [1, '#FF0000']]
colors = {
    'black': '#000000',
    'changes': '#2420AC',
    'gt_true':'#1a92c6',
    'gt_false':'#cce5ff',
    'pr_true':'#ff8000',
    'pr_false':'#ffe5cc',
    'heatmap_font_1':'#3c3636',
    'heatmap_font_2':'#efecee'
}


# ============================ Defs ===============================

def get_arguments():
    parser = argparse.ArgumentParser(description='Demo for skill-assessment')
    parser.add_argument('--root_dir', type=str, default= "results", help='dir for args')
    parser.add_argument('--result_dir', type=str, default='demo/results', help='results file name')
    parser.add_argument('--datalist_dir', type=str, default='data/BEST/new_splits', help='datalist path')
    parser.add_argument('--debug', action='store_true', help='debug True or not')
    return parser.parse_args()

# Get Video Path Lists
def load_path(task):
    vid_list = [x.strip().split(' ') for x in open(os.path.join(os.getcwd(), args.datalist_dir, task, "test.txt"))]
    return vid_list

# Load Video
def load_video(task, vid_name, origin = False):
    if origin:
        video_path=glob.glob(os.path.join(VIDEOS_PATH ,'demo/videos', task, vid_name))
    else:
        video_path=glob.glob(os.path.join(VIDEOS_PATH, args.result_dir, args.arg, "lap_"+args.lap, task, "video_att", vid_name))
    return os.path.join(video_path[0]+"/")

# Load Results
def load_result(task, vid_name, pos_neg, params):
    # Load the dataframe
    if params.rank_aware_loss:
        csv = "best_epoch_p_att.csv"
    else:
        csv = "best_epoch_att.csv"
    data_path = os.path.join(args.result_dir, args.arg, "lap_"+args.lap, task, csv)
    videos_df = pd.read_csv(data_path)
    video_df = videos_df[videos_df["Unnamed: 0"] == vid_name][0:1]
    att_map = video_df[list(str(i) for i in range(1,401))].values
    att_map = normalize_heatmap(att_map)
    return att_map

def normalize_heatmap(x):
    # choose min (0 or smallest scalar)
     min = x.min()
     max = x.max()
     result = (x-min)/(max-min)
     return result

# ============================ Layouts ===============================

# Main App
app.layout = html.Div([
    # ==== Banner display ====
    html.Div([
        html.H2(
            'Skill Assessment',
            id='title',
            style={
                'width': '20%',
                'margin-left': '2%',
                'height': '110px',
                'display': 'inline-block', 
                'vertical-align': 'top'
                }
        ),
        html.H2(
            'Aoki lab.　Takuya Kobayashi',
            id='name',
            style={
                'width': '65%',
                'margin-top': '3%',
                'font-size': '1em',
                'display': 'inline-block', 
                'vertical-align': 'top'
                }
        ),
        html.Img(
            src="https://wwwdc05.adst.keio.ac.jp/kj/vi/common/img/thumbF2.png",
            style={
                'width': '10%',
                'margin-top': '1%',
                'height': '60px',
                'display': 'inline-block', 
                'vertical-align': 'top'
                }
        )
    ],
        className="banner",
        style={
                'background-color': '#66b2ff',
                'height': '75px',
                'margin-bottom': '10px'
            }
                 
    ),

    # ==== Body ====
    html.Div([
        # === Top sides Components ===
        html.Div([
            # == Left side Components ==
            html.Div([
                # = Video Player =
                html.Div([
                    html.Img(
                        style={'width': '100%', 
                                'height': '300px',#400,
                                'margin': '0px 0px 0px 0px'
                                },
                        id="images",
                        ),
                    #  frame type
                    html.Div(
                        id='image-type',
                        style={'display': 'none'}
                        )
                    ],
                    id='video-player',
                    style={'color': 'rgb(255,255,255)',
                           'width': '70%',
                           'display': 'inline-block',
                           'vertical-align': 'top'
                           }
                ),
                # = Chose Play Mode = 
                html.Div([
                    html.Div([
                        html.Div([
                                "Image Display"
                                ],
                                 style={'margin': '0px 0px 0px 0px',
                                    'text-align': 'center'}
                                ), 
                        html.Div([
                                daq.ToggleSwitch(
                                    label='Attention : Origin',
                                    labelPosition='bottom',
                                    id='image-toggle',
                                    value=True
                                    )
                                ],
                                className='image mode toggle',
                                style={"margin-top" : "20px"}
                                )
                    ],
                    className='image mode',
                    style={'background-color': '#99CCFF'}
                    ),
                    html.Div([
                        "Test"
                    ],
                    className='video result',
                    style={'background-color': '#CCFFCC'}
                    )
                ],
                className='play mode',
                style={'width': '30%',
                       'display': 'inline-block',
                       'vertical-align': 'top'
                       }                
                )
            ],
            className='left component',
            style={'width': '60%', 
                   'display': 'inline-block', 
                   'vertical-align': 'top'}
            ),
            
            # == Right side Components ==
            html.Div([
                # Video selection
                html.Div([
                    html.Div([
                            "Video Selection"
                             ],
                             style={'margin': '0px 0px 0px 0px',
                                    'text-align': 'center'}
                             ),       
                    # task
                    html.Div(["Task",
                               dcc.Dropdown(
                                 options=[
                                    {'label':'Apply_eyeliner','value':'apply_eyeliner'},
                                    {'label':'Braid_hair','value':'braid_hair'},
                                    {'label':'Origami','value':'origami'},
                                    {'label':'Scrambled_eggs','value':'scrambled_eggs'},
                                    {'label':'Tie_tie','value':'tie_tie'}
                                 ],
                                 value='apply_eyeliner',
                                 id='task-selection',
                                 clearable=False,
                                 style={'color': colors['black']
                                        }
                                 )
                             ],
                             style={'margin': '10px 0px 20px 25px',
                                    'width': '30%',
                                    'display': 'inline-block', 
                                    'vertical-align': 'middle',
                                    'color': colors['changes']},
                             className='task'
                             ),
                    # videoset 
                    html.Div(["Video Pair",
                              dcc.Input(
                                value=1,
                                id='pair-selection',
                                type='number',
                                placeholder='Video Num',
                                style={'width': 85,
                                       'height': 35,
                                       'test-align': 'center'
                                       }
                                )
                             ],
                             style={'margin': '21px 0px 35px 30px',
                                    'width': '27%',
                                    'display': 'inline-block', 
                                    'vertical-align': 'middle',
                                    'color': colors['changes']},
                             className='videopair'
                             ),
                    # pos or neg
                    html.Div(["Pos Neg",
                              dcc.RadioItems(
                                options=[
                                    {'label':'Positive','value':0},
                                    {'label':'Negative','value':1}
                                ],
                                value=0,
                                id='pos_neg-selection',
                                labelStyle={'display': 'block'}
                                )
                             ],
                             style={'margin': '20px 0px 25px 0px',
                                    'width': '30%', 
                                    'display': 'inline-block', 
                                    'vertical-align': 'middle',
                                    'color': colors['changes']},
                             className='pos_neg'
                             ),
                    html.Div([
                              html.Button(
                                   id="submitvideo-button", 
                                   n_clicks=0, 
                                   children="Submit",
                                   style={'margin-left': '42%',
                                          'width':'80px',
                                          'height':'40px'}
                               )
                             ],
                             style={'vertical-align': 'middle',
                                    'color': colors['changes'],
                                    'margin-bottom': '0px'},
                             className='submit button'
                             )

                    ],
                    style={'margin-bottom': '0px',
                           'background-color': '#20B2AA'
                           },
                    className='video selection'
                    ),

                # = Chose Play Mode = 
                html.Div([
                    html.Div([
                            "Play Mode"
                             ],
                             style={'margin': '0px 0px 0px 0px',
                                    'text-align': 'center'}
                             ), 
                    html.Div([
                            daq.ToggleSwitch(
                                label='Auto : Manual',
                                labelPosition='bottom',
                                id='mode-toggle',
                                value=True
                                )
                            ],
                            className='play mode toggle',
                            style={"margin-top" : "20px"}
                            )
                    ],
                    className='play mode',
                    style={"margin-top" : "0px",
                           "margin-bottom" : "10px",
                           'background-color': '#7B68EE',
                           'width': '40%',
                           'display': 'inline-block',
                           'vertical-align': 'top',
                           }
                    ),
                
                # Chose Trained Data
                html.Div([
                    html.Div([
                            "Model Weight"
                             ],
                             style={'margin': '0px 0px 0px 0px',
                                    'text-align': 'center'}
                             ),  
                    html.Div([dcc.Dropdown(
                                 options=[
                                    {'label': label, 'value': i} for i, label in enumerate(arglaplist)
                                    ],
                                 value=0,
                                 id='weight-selection',
                                 clearable=False,
                                 style={'color': colors['black'],
                                        'margin-top': '20px',
                                        'margin-bottom': '20px',
                                        'margin-left': '15px',
                                        'width': '90%'
                                        }
                                 )
                             ],
                             className='weight dropdown',
                             style={'margin': '0px 0px 0px 0px',
                                    'color': colors['changes']}
                             ),                   
                    ],
                    className='weight model choose',
                    style={"margin-top" : "0px",
                           "margin-bottom" : "0px",
                           'background-color': '#2F4F4F',
                           'width': '60%',
                           'display': 'inline-block',
                           'vertical-align': 'top'
                           }
                    )
                ],
                className="right components",
                style={'margin-left': '2%',
                       'width': '37%', 
                       'display': 'inline-block', 
                       'vertical-align': 'top'}
                )
            ],   
            className="top components",
            style={'margin-left': '5%',
                   'margin-bottom': '0px'
                }
            ),


        # === Bottom sides Components ===
        html.Div([
            dcc.Graph(
                style={
                    'width' : '100%'
                    },
                id="heatmap-confidence"
                ),
            html.Div([
                dcc.Slider(
                    min=0,
                    max=video_length-1,
                    marks={i: 't:{}'.format(i) for i in range(0,video_length+1,20)},
                    value=0,
                    updatemode='mouseup',
                    id='slider-frame-position',
                    )
                ],
                style={"margin-top":"-90px",
                       "margin-left":"56px",
                       "width":"78%"}
                ),
            dcc.Interval(
                id='interval-component',
                n_intervals=0,
                max_intervals=video_length
                )                
            ], 
            className="bottom components",
            style={
                   'margin-left': '5%'}
            )
        
        
        ], 
        className="body"
        )


    ],
    className="all"
    )



# ============================ Operations ===============================

# Data Loading
@app.server.before_first_request
def load_all_videos():
    print("start")


# Images Display
@app.callback(Output("images", "src"),
             [Input("submitvideo-button", "n_clicks"),
              Input("slider-frame-position", "value"),
              Input("image-type", "children")],
             [State("task-selection", "value"),
              State("pair-selection", "value"),
              State("pos_neg-selection", "value")])
def update_image_src(n_clicks, frame, att, task, pair, pos_neg):
    vid_list = load_path(task)
    video_path = load_video(task, vid_list[pair][pos_neg], origin=att)
    frame_name = str(frame+1).zfill(5) + ".png"
    return video_path + frame_name

@app.server.route('{}<path:image_path>.png'.format(VIDEOS_PATH))
def serve_image(image_path):
    img_name = '{}.png'.format(image_path)
    return flask.send_file(VIDEOS_PATH + img_name)


# Spatial Attention boolian
@app.callback(Output("image-type", "children"),
             [Input("weight-selection", "value"),
               Input("image-toggle", "value")])
def update_spatial_attention(value, origin):  
    args.arg = arglist[value]
    args.lap = laplist[value]
    params = Dict(yaml.safe_load(open(os.path.join('args',args.arg+'.yaml'))))
    if not params.spatial_attention or origin:
        return True
    else:
        return False


# Temporal Attention display
@app.callback(Output("heatmap-confidence", "figure"),
             [Input("submitvideo-button", "n_clicks"),
              Input("weight-selection", "value")],
             [State("task-selection", "value"),
              State("pair-selection", "value"),
              State("pos_neg-selection", "value")])
def update_temporal_attention(n_clicks, value, task, pair, pos_neg):
    params = Dict(yaml.safe_load(open(os.path.join('args',args.arg+'.yaml'))))
    vid_list = load_path(task)
    vid_name = vid_list[pair][pos_neg]
    att_data = load_result(task, vid_name, pos_neg, params)

    return {'data': [{
                'z': att_data.tolist(),
                'y': [""],
                'ygap': 1,
                'reversescale': 'true',
                'colorscale': colorscale,
                'type': 'heatmap',
                'showscale': True
                }],
            'layout': {
                'height': 300,
                'width': 1189,
                'xaxis': {'side':'top'},
                }
            }


# Auto mode
@app.callback(Output("slider-frame-position", "value"),
              Input("interval-component", "n_intervals"),
              State("slider-frame-position", "value"))
def refresh_frame_select(n_intervals, frame):
    return min(frame+2, video_length-1)


# Toggle Auto and Manual mode
@app.callback(Output('interval-component', 'interval'),
              Input("mode-toggle", "value"))
def manual_mode(value):
    if value:
        return 24*60*60*1000
    else:
        return 800


# ============================ Others ===============================

external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",  # Normalize the CSS
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto"  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    "https://cdn.rawgit.com/plotly/dash-object-detection/875fdd6b/custom-styles.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

if __name__ == '__main__':
    global args, params
    args = get_arguments()
    args.arg = arglist[0]
    args.lap = laplist[0]
    params = Dict(yaml.safe_load(open(os.path.join(args.root_dir, 'arg.yaml'))))
    # Run
    app.run_server(debug=args.debug, host='0.0.0.0', port=8888)