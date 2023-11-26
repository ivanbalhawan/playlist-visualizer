import json
from typing import Any
from typing import Dict
from typing import List

import pandas as pd
import plotly.express as px
import requests
from dash import Dash
from dash import Input
from dash import Output
from dash import callback
from dash import dcc
from dash import html

from spotify_api import audio_features

default_columns = ["track_id", "album_name", "track_name"] + audio_features
default_data_json = pd.DataFrame(columns=default_columns).to_json(orient="records")

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1(
            id="album-name-header",
            children="Did not Fetch",
            style={"textAlign": "center"},
        ),
        html.Button("Get Favorite Tracks Features!", id="fetch-album-button"),
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
)
def fetch_album_features(button_click):
    url: str = "http://localhost:56626/tracks/saved"
    response = requests.get(url)
    data: List[Dict[str, Any]] = response.json()
    album_data: pd.DataFrame = pd.DataFrame(data)

    album_data_json = album_data.to_json(orient="records")
    return "Fetched Tracks", album_data_json


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
    fig = px.scatter(album_features_df, x=feature_x, y=feature_y, hover_name="track_name")
    return fig


if __name__ == "__main__":
    app.run(debug=True)
