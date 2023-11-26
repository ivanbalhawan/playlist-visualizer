import os

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

ACCOUNTS_BASE_URL = "https://accounts.spotify.com"
API_BASE_URL = "https://api.spotify.com"

audio_features = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]


def get_auth_token():
    # request an access token
    url = f"{ACCOUNTS_BASE_URL}/api/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {
        "grant_type": "client_credentials",
        "client_id": os.environ.get("CLIENT_ID"),
        "client_secret": os.environ.get("CLIENT_SECRET"),
    }
    response = requests.post(url, headers=headers, data=payload)
    auth_token = response.json().get("access_token")
    return auth_token


def get_album_tracks(album_id, auth_token):
    url = f"{API_BASE_URL}/v1/albums/{album_id}/tracks"
    headers = {
        "Authorization": f"Bearer {auth_token}",
    }
    response = requests.get(url, headers=headers)
    items = response.json().get("items", [])

    tracks = []
    for item in items:
        track = {
            "track_id": item.get("id"),
            "track_name": item.get("name"),
        }
        tracks.append(track)
    album_tracks_df = pd.DataFrame(tracks)
    return album_tracks_df


def get_tracks_audio_features(track_id_list, auth_token):
    url = f"{API_BASE_URL}/v1/audio-features"
    headers = {"Authorization": f"Bearer {auth_token}"}
    params = {"ids": ",".join(track_id_list)}
    response = requests.get(url, headers=headers, params=params)
    audio_features_response = response.json().get("audio_features")
    audio_features_df = pd.DataFrame(audio_features_response)
    audio_features_df.rename(
        columns={"id": "track_id"},
        inplace=True,
    )
    return audio_features_df


def get_album_features(album_id):
    auth_token = get_auth_token()
    album_details_url = f"{API_BASE_URL}/v1/albums/{album_id}"
    headers = {"Authorization": f"Bearer {auth_token}"}

    album_details_response = requests.get(album_details_url, headers=headers)
    album_name = album_details_response.json().get("name", "Album Name")

    album_tracks_df = get_album_tracks(album_id, auth_token)
    album_tracks_df["album_name"] = album_name

    album_tracks_audio_features_df = get_tracks_audio_features(
        track_id_list=album_tracks_df.track_id.to_list(),
        auth_token=auth_token,
    )

    album_df = pd.merge(
        album_tracks_df,
        album_tracks_audio_features_df,
        on="track_id",
        how="left",
    )

    return album_df


if __name__ == "__main__":
    auth_token = get_auth_token()
    album_features_df = get_album_features("1U0A6RPNJB05PtuBcaTM7o", auth_token)
    album_features_df.to_pickle("album_features_df.pickle")
