import dash
import plotly.express as px
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_html_components as html
from sqlite_db import SQLDb
import math
from sklearn.metrics import accuracy_score
from plotly.graph_objects import Scatter3d
import random


def get_plotly_fig(pandas_df, original_df):
    fig = px.scatter_3d(pandas_df, x="X", y="Y", z="Z",
                        color="ORIGINAL_LABEL",
                        color_discrete_sequence=color_sequence,
                        hover_data={
                            'X': False,
                            'Y': False,
                            'Z': False,
                            'DATAPOINT_NAME': True,
                            'ORIGINAL_LABEL': True,
                            'CLASSIFIED_AS_LABEL': True
                        },
                        range_x=[math.floor(original_df["X"].min()),
                                 math.ceil(original_df["X"].max())],
                        range_y=[math.floor(original_df["Y"].min()),
                                 math.ceil(original_df["Y"].max())],
                        range_z=[math.floor(original_df["Z"].min()),
                                 math.ceil(original_df["Z"].max())],

                        )
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(scene_aspectmode='cube')
    fig.update_layout(
        {'height': 1000}
    )
    return fig


def get_accuracy(my_df):
    acc = accuracy_score(my_df["ORIGINAL_LABEL"], my_df["CLASSIFIED_AS_LABEL"])
    return acc * 100


def get_uid_xyz(input_df, uid):
    indexes = input_df.loc[input_df['DATAPOINT_NAME'] == uid].index.values.astype(int)
    if len(indexes) == 0:
        return None, None, None
    index = indexes[0]
    return input_df["X"][index], input_df["Y"][index], input_df["Z"][index]


if __name__ == "__main__":
    r = lambda: random.randint(0, 255)
    sql_db = SQLDb("test", create_table=False)
    # This is a very specific test
    # TODO: This should be modified to a more generic one
    df = sql_db.get_pandas_frame(sql_db.get_all_table_names()[0])
    unique_colors = len(df["ORIGINAL_LABEL"].unique())
    color_sequence = ['#%02X%02X%02X' % (r(), r(), r()) for _ in range(unique_colors)]
    fig = get_plotly_fig(df, df)
    fig.layout.uirevision = True

    options = [
        {'label': k, 'value': k} for k in df["ORIGINAL_LABEL"].unique()
    ]
    dropdown_menu_items = [
        {'label': k, 'value': k} for k in sql_db.get_all_table_names()
    ]
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
    app.layout = dbc.Container(
        [
            dbc.Row([
                dbc.Col([
                    html.H2("Manifolk: 3D TSNE graphs for every epoch using Plotly, Dash and MySQLite"),
                    html.A("(Source code)", href="https://github.com/krishnachaitanya7/Manifolk")
                ], width=True),
            ], align="end"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Select the table you want to plot", style={"text-decoration": "underline"}),
                        dcc.Dropdown(
                            id='select_table',
                            options=dropdown_menu_items,
                            value=dropdown_menu_items[0]['value']
                        ),
                        html.H5("Accuracy Under the current Settings", style={"text-decoration": "underline"}),
                        html.Strong(id='accuracy_info', style={'whiteSpace': 'pre-line'}),
                        html.H5("Mark Unique Datapoint (Optional)", style={"text-decoration": "underline"}),
                        dbc.Input(
                            id="input_UID",
                            type="text",
                            placeholder="Input DATAPOINT_NAME",
                        ),
                        dbc.FormText("Input the point's DATAPOINT_NAME you want to track"),
                        dbc.FormFeedback(
                            "That looks like a valid point present in the graph :)", valid=True
                        ),
                        dbc.FormFeedback(
                            "Sorry, either the DATAPOINT_NAME is not present in the current setting, or it's not there in the database at all",
                            valid=False,
                        ),
                        html.H5("Select Labels to be plotted:", style={"text-decoration": "underline"}),
                        dbc.Checklist(
                            id="checklist",
                            options=options,
                            value=[options[0]['value']],
                            labelStyle={'display': 'inline-block'},
                            labelCheckedStyle={"color": "green"},
                        )
                    ])
                ], width=3),
                dbc.Col([
                    dcc.Graph(
                        id="tsne_figure",
                        figure=fig),
                    dcc.Slider(
                        id="epoch_slider",
                        min=df["epoch"].min(),
                        max=df["epoch"].max(),
                        value=df["epoch"].max(),
                        marks={f'{df["epoch"].min()}': 'Epoch:'},
                        tooltip={'always_visible': True, 'placement': 'bottom'}
                    ),
                ], width=True)
            ]),

        ],
        fluid=True
    )


    @app.callback(
        [Output('tsne_figure', 'figure'), Output('accuracy_info', 'children'), Output("input_UID", "valid"),
         Output("input_UID", "invalid")],
        [Input('checklist', 'value'), Input('tsne_figure', 'relayoutData'), Input('epoch_slider', 'value'),
         Input("input_UID", "value"), Input('select_table', 'value')])
    def update_figure(*vals):
        # selected_labels, data, value, value_uid
        selected_labels = vals[0]
        data = vals[1]
        slider_value = vals[2]
        datapoint_uid = vals[3]
        df = sql_db.get_pandas_frame(vals[4])
        filtered_df = df[df["ORIGINAL_LABEL"].isin(selected_labels)]
        filtered_df = filtered_df.loc[filtered_df["epoch"] == slider_value]
        accuracy = get_accuracy(filtered_df)
        fig = get_plotly_fig(filtered_df, df)
        fig.update_layout(scene_camera=data['scene.camera'])
        if datapoint_uid:
            point_x, point_y, point_z = get_uid_xyz(filtered_df, datapoint_uid)
            if point_x and point_y and point_z:
                fig.add_trace(Scatter3d(x=[point_x], y=[point_y], z=[point_z],
                                        mode='markers', marker=dict(symbol='diamond')))
                set_radio_button = "YES"
            else:
                # The point doesn't exist, show it to user
                set_radio_button = "NO"
        else:
            set_radio_button = "NO"

        return fig, f"{accuracy:.2f}%", set_radio_button == "YES", set_radio_button != "YES"


    app.run_server(debug=False)
