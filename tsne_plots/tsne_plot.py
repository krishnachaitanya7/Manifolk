# references: https://plotly.com/python/sliders/
import dash
import plotly.express as px
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_html_components as html
from sqlite_db import SQLDb
import math
from plotly.graph_objects import Scatter3d


def get_plotly_fig(pandas_df, original_df):
    fig = px.scatter_3d(pandas_df, x="X", y="X", z="Z", animation_frame="epoch",
                        color="ORIGINAL_LABEL",
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
    # fig.add_trace(Scatter3d(x=[pandas_df.iloc[0]["X"]], y=[pandas_df.iloc[0]["Y"]], z=[pandas_df.iloc[0]["Z"]],
    #                         mode='markers', marker=dict(symbol='cross')))
    # fig.update_xaxes(autorange=False)
    # fig.update_yaxes(autorange=False)
    # fig.update_zaxes(autorange=False)
    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(scene_aspectmode='cube')
    fig.update_layout(
        {'height': 1000}
    )
    return fig


if __name__ == "__main__":
    sql_db = SQLDb("test", create_table=False)
    # This is a very specific test
    # TODO: This should be modified to a more generic one
    df = sql_db.get_pandas_frame("[mnist_2020-11-08-22:59:17]")
    fig = get_plotly_fig(df, df)

    options = [
        {'label': k, 'value': k} for k in df["ORIGINAL_LABEL"].unique()
    ]
    app = dash.Dash()
    app.layout = html.Div(children=[
        dcc.Graph(
            id="tsne_figure",
            figure=fig),
        # dropdown
        html.P([
            html.Label("Select Labels"),
            dcc.Checklist(
                id="checklist",
                options=options,
                value=[options[0]['value']],
                labelStyle={'display': 'inline-block'}
            )
        ])

    ])


    @app.callback(
        Output('tsne_figure', 'figure'),
        [Input('checklist', 'value'), Input('tsne_figure', 'relayoutData')])
    def update_figure(selected_labels, data):
        filtered_df = df[df["ORIGINAL_LABEL"].isin(selected_labels)]
        fig = get_plotly_fig(filtered_df, df)
        fig.update_layout(scene_camera=data['scene.camera'])
        fig.update_layout(transition_duration=500)
        return fig


    app.run_server(debug=True)
