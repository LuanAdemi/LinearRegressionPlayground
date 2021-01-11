import base64
import datetime
import io
import time

import pandas as pd
import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State

from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

import numpy as np

m = 0
c = 0
df = []

fig1 = []
fig2 = []

xMSE = []
yMSE = []
zMSE = []


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css', 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css']
external_scripts = ['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML']

mathjaxMap = {
    "MSE":"\\[ L ={\\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2} \\]"
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=external_scripts)

app.layout = html.Div(style={'height':'100%'},children=[
    dcc.Store(id="store"),
    html.Div(id="foo",children=[]),

    html.Div(style={"height":"55px", "padding":"10px", "background-color":"white", "box-shadow":"0px 2px 5px lightgrey","position":"absolute","z-index":"10","width":"100%"},children=[
        html.Button(className="bubbly-button",id='exitButton', n_clicks=0, style={"float":"left","height":"100%", "margin-right":"20px","background-color":"black","color":"white","border":"none"}, children=[
            html.I(className="fa fa-chevron-left", style={"float":"left","lineHeight":"55px"}),
            html.P('Exit', style={"float":"left","lineHeight":"55px","margin-left":"10px"})
        ]),
        html.Button(className="bubbly-button",id='runButton', n_clicks=0, style={"float":"left","height":"100%", "margin-right":"20px","background-color":"#1940FF","color":"white","border":"none"}, children=[
            html.I(className="fa fa-play", style={"float":"left","lineHeight":"55px"}),
            html.P('Run', style={"float":"left","lineHeight":"55px","margin-left":"10px"})
        ]),
        html.Button(className="bubbly-button",id='stepButton', n_clicks=0, style={"float":"left","height":"100%", "margin-right":"20px","background-color":"white","color":"#1940FF","border":"2px solid #1940FF"}, children=[
            html.I(className="fa fa-forward", style={"float":"left","lineHeight":"55px"}),
            html.P('Step', style={"float":"left","lineHeight":"55px","margin-left":"10px"})
        ]),
        html.H5("1.1 Linear Regression",style={"position":"absolute", "lineHeight":"40px","width":"100%","text-align":"center","z-index":"-10"}),

        html.A(

            html.Button(className="bubbly-button", id='github', n_clicks=0, style={"float":"right","height":"100%", "margin-right":"20px","background-color":"white","color":"black","border":"2px solid black"}, children=[
                html.I(className="fa fa-github fa-2x", style={"float":"left","lineHeight":"55px"}),
                html.P('GitHub', style={"float":"left","lineHeight":"55px","margin-left":"10px"})
            ])

        , href="https://github.com/LuanAdemi"),
        html.A(
        html.Button(className="bubbly-button", id='learn', n_clicks=0, style={"float":"right","height":"100%", "margin-right":"20px","background-color":"white","color":"#1940FF","border":"2px solid #1940FF"}, children=[
            html.I(className="fa fa-book", style={"float":"left","lineHeight":"55px", "font-size":"15px"}),
            html.P('Learn more', style={"float":"left","lineHeight":"55px","margin-left":"10px"})
        ])

        , href="https://en.wikipedia.org/wiki/Gradient_descent"),






    ]),
    # body
    html.Div(id='body', style={'background-color': '#F2F3F4', 'height':'100%', "position":"absolute", "width":"100%","margin-top":"70px"}, children=[

        # parameters
        html.Div(id='parameters', style={'background-color': '#F9F9F9', "width": "450px", 'height':'100%', "float":"left"}, children=[

            html.Div(id='parameters_inner', style={'padding': '10px'}, children=[


                html.H6("Optimizer",id="label", style={'text-align': 'left'}),

                dcc.Dropdown(id="slct_optimizer",
                             options=[
                                 {"label": "SGD", "value": "SGD"}],
                             multi=False,
                             value="SGD",
                             style={'width': "100%"}
                             ),

                html.Br(),
                html.H6("Loss", style={'text-align': 'left'}),

                dcc.Dropdown(id="slct_loss",
                             options=[
                                 {"label": "MSE", "value": "MSE"}],
                             multi=False,
                             value="MSE",
                             style={'width': "100%"}
                             ),

                #html.P('\\[ L ={\\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2} \\]'),
                html.Br(),
                html.H6("Iterations", style={'text-align': 'left'}),

                dcc.Slider(
                    id='iters',
                    min=0,
                    max=100,
                    step=None,
                    marks={
                        1:'1',
                        5:'5',
                        10:'10',
                        50:'50',
                        100:'100'
                    },
                    value=10,
                    ),

                html.Br(),
                html.H6("Learningrate", style={'text-align': 'left'}),

                dcc.Slider(
                    id='lr',
                    min=1e-7,
                    max=1e-4,
                    step=None,
                    marks={
                        1e-7:'1e-7',
                        1e-6:'1e-6',
                        1e-5:'1e-5',
                        1e-4:'1e-4'
                    },
                    value=1e-5,
                    ),
                html.Br(),
                html.H6("Dataset", style={'text-align': 'left'}, id="dataname"),

                dcc.Loading(
                            id="loading2",
                            style={"margin-top":"260px"},
                            children=[html.Div(id='output-data-upload',style={"max-height":"450px"})],
                            type="default"

                        ),


                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                    ]),
                    style={
                        'width': '100%',
                        'height': '350px',
                        'lineHeight': '350px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        },
                    # Allow multiple files to be uploaded
                    multiple=False
                ),




            ]),
        ]), # parameters

        # main plots

            html.Div(style={"height":"100%", "width":"calc( 100% - 450px)", "position":"absolute", "margin-left":"450px","margin-top":"20px","display":"none"}, id="dashboard", children=[
                html.Div([
                    # Loss-Gradient
                    html.Div(children=[
                        dcc.Graph(
                        id='MSE_GV',
                        style={'height': '100%', "width": "100%"}
                    )

                    ], style={"height": "100%", "width":"50%","float":"left"}),

                    # Result
                    html.Div(children=[

                        dcc.Graph(
                        id='Data',
                        style={'height': '50%', "width": "calc( 100% - 20px)"}
                    ),

                        dcc.Graph(
                        id='Loss',
                        style={'height': '50%', "width": "calc( 100% - 20px)"}
                    ),



                    ], style={"height": "100%", "width":"50%","float":"left"}),
                    # Loss-Plot

                ], style={"height":"calc( 100%  - 150px)"})



            ])



    ]) # body


])

def MSE(X,Y,M,C):
    y = M.copy()
    n = len(X)
    for i,m in enumerate(M):
        for j,c in enumerate(C):
            y[i][j] = (1/n) * np.sum((Y - (m*X+c))**2, axis=0)
    return y

def MSE_s(X,Y,m,c):
    n = len(X)
    return (1/n) * np.sum((Y - (m*X+c))**2, axis=0)

def parse_contents_as_plot(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file. Make sure it is the right format (.csv or .xls)'
        ])

    fig = px.scatter(df, x="X", y="Y", width=430, height=380)
    fig.update_layout(paper_bgcolor = '#F9F9F9')

    return html.Div([ dcc.Graph(id='example-graph', figure=fig)])

def storeData(contents, filename, date):


    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file. Make sure it is the right format (.csv or .xls)'
        ])

    m = np.random.randint(-10,10)
    c = np.random.randint(-10,10)
    xMSE = np.outer(np.linspace(-10, 10, len(df["X"])), np.ones(len(df["X"])))
    yMSE = xMSE.copy().T
    zMSE = MSE(df["X"],df["Y"],xMSE,yMSE)

    loss = []
    loss.append(MSE_s(df["X"],df["Y"],m, c))

    return  df.to_dict("rows"), m, c, zMSE, loss

def SGDStep(df, m, c, lr):
    L = lr
    X = df['X']
    Y = df['Y']
    n = float(len(X))
    Y_pred = m*X + c

    D_m = (-2/n) * sum(X * (Y-Y_pred))
    D_c = (-2/n) * sum(Y-Y_pred)

    m -= L * D_m
    c -= L * D_c
    return m, c

def bakePlots(df, m, c, zMSE, loss):
    xMSE = np.outer(np.linspace(-10, 10, len(df["X"])), np.ones(len(df["X"])))
    yMSE = xMSE.copy().T

    x1 = np.linspace(df["X"].min(),df["X"].max(), len(df["X"]))

    fig1 = go.Figure()
    fig1.add_trace(go.Surface(x=xMSE,y=yMSE,z=zMSE, name="Loss"))
    fig1.add_trace(go.Scatter3d(x=[m], y=[c], z=[MSE_s(df["X"],df["Y"],m,c)],
                                   mode='markers',name="Gradient"))
    fig1.update_layout(
        paper_bgcolor = '#F2F3F4',
        title="Loss - Gradient Visualisation",
        uirevision='constant'
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["X"], y=df["Y"], mode='markers',name="Data"))
    fig2.add_trace(go.Scatter(x=x1, y=(m*x1+c), mode='lines',name=f'f(x)={round(m,2)}x+{round(c,2)}'))
    fig2.update_layout(paper_bgcolor = '#F2F3F4', title="Result", uirevision='constant')

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=[i for i in range(len(loss))], y=loss, mode='lines',name="Loss"))
    fig3.update_layout(paper_bgcolor = '#F2F3F4', title="Loss", uirevision='constant')
    return fig1, fig2, fig3

# creates and displays a plot object created from the uploaded data, hides to file uploader and stores the data in an invisible div
@app.callback(Output('output-data-upload', 'children'),
              Output('upload-data', 'style'),
              Output('dataname', "children"),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def makePlot(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        return parse_contents_as_plot(list_of_contents, list_of_names, list_of_dates), {"display":"none"}, f"Dataset ({list_of_names})"
    else:
        raise PreventUpdate

@app.callback(Output('dashboard', "style"),
              Output('MSE_GV', 'figure'),
              Output('Data', 'figure'),
              Output('Loss', 'figure'),
              Output("store", "data"),
              Input("runButton",'n_clicks'),
              Input('upload-data', 'contents'),
              State('iters', 'value'),
              State('lr', 'value'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
              State("store", "data"),
              State('MSE_GV', 'figure'),
              State('Data', 'figure'),
              State('Loss', 'figure'))
def dashboard(n_clicks, list_of_contents, iters, lr, list_of_names, list_of_dates, data, fig1, fig2, fig3):
    if data is None and list_of_contents is not None:
        sData = storeData(list_of_contents, list_of_names, list_of_dates)
        df = pd.DataFrame.from_dict(sData[0])
        m = sData[1]
        c = sData[2]
        zMSE = sData[3] # stored due to havy computing time
        loss = sData[4]
        fig1, fig2, fig3 = bakePlots(df, m, c, zMSE, loss)
        return {"height":"100%", "width":"calc( 100% - 450px)", "position":"absolute", "margin-left":"450px","margin-top":"20px","display":"block"}, fig1, fig2, fig3, sData

    elif data is not None and fig1 is None:
        df = pd.DataFrame.from_dict(data[0])
        m = data[1]
        c = data[2]
        zMSE = data[3] # stored due to havy computing time
        loss = data[4]

        fig1, fig2, fig3 = bakePlots(df, m, c, zMSE, loss)
        data = (df.to_dict("rows"), m,c,zMSE, loss)
        return {"height":"100%", "width":"calc( 100% - 450px)", "position":"absolute", "margin-left":"450px","margin-top":"20px","display":"block"}, fig1, fig2, fig3, data

    elif data is not None and fig1 is not None:
        df = pd.DataFrame.from_dict(data[0])
        m = data[1]
        c = data[2]
        for _ in range(iters):
            m,c = SGDStep(df, m, c, lr)
        zMSE = data[3] # stored due to havy computing time
        loss = data[4]

        fig1, fig2, fig3 = bakePlots(df, m, c, zMSE, loss)
        loss.append(MSE_s(df["X"],df["Y"],m, c))
        data = (df.to_dict("rows"), m,c,zMSE, loss)
        return {"height":"100%", "width":"calc( 100% - 450px)", "position":"absolute", "margin-left":"450px","margin-top":"20px","display":"block"}, fig1, fig2, fig3, data

    else:
        raise PreventUpdate





if __name__ == '__main__':
    app.run_server(debug=False)
