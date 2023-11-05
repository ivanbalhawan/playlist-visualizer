import json

from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import pandas as pd

from spotify_api import audio_features
from spotify_api import get_album_features

default_columns = ["track_id", "album_name", "track_name"] + audio_features
default_data_json = pd.DataFrame(columns=default_columns).to_json(orient="records")

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            id="album-name-header",
            children="Album Name",
            style={"textAlign": "center"},
        ),
        dcc.Input(id="album-id-state", placeholder="Album ID", type="text", value=""),
        html.Button("Get Album Features!", id="fetch-album-button"),
        dcc.Dropdown(audio_features, audio_features[0], id="dropdown-x"),
        dcc.Dropdown(audio_features, audio_features[1], id="dropdown-y"),
        dcc.Graph(id="graph-content"),
        dcc.Store(id="album-data", data=default_data_json),
        dcc.Store(
            id="features-data",
            data={"feature_x": audio_features[0], "feature_y": audio_features[1]},
        ),
    ]
)


@callback(
    Output("album-name-header", "children"),
    Output("album-data", "data"),
    Input("fetch-album-button", "n_clicks"),
    State("album-id-state", "value"),
)
def fetch_album_features(button_click, album_id_state):
    default_dataframe = pd.DataFrame(columns=default_columns)
    if album_id_state == "":
        return "Album Name", default_dataframe.to_json(orient="records")
    album_data = get_album_features(album_id_state)
    album_name = album_data.album_name.values[0]
    album_data_json = album_data.to_json(orient="records")
    return album_name, album_data_json


@callback(
    Output("features-data", "data"),
    Input("dropdown-x", "value"),
    Input("dropdown-y", "value"),
)
def update_features_dict(feature_x, feature_y):
    features_dict = {"feature_x": feature_x, "feature_y": feature_y}
    return json.dumps(features_dict)


@callback(
    Output("graph-content", "figure"),
    Input("album-data", "data"),
    Input("features-data", "data"),
)
def plot_new_dataframe(album_data_json, features_dict_json):
    album_features_df = pd.read_json(album_data_json, orient="records")
    features_dict = json.loads(features_dict_json)
    feature_x = features_dict.get("feature_x", audio_features[0])
    feature_y = features_dict.get("feature_y", audio_features[1])
    fig = px.scatter(album_features_df, x=feature_x, y=feature_y)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
