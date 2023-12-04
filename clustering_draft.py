from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List


import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
import tqdm


relevant_features = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "speechiness",
    "valence",
]

def dbscan_clustering(
    X: np.ndarray,
    eps: float,
    min_samples: int,
):
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    ).fit_predict(X)
    return labels


def meanshift_clustering(
    X: np.ndarray,
):
    labels = MeanShift().fit_predict(X)
    return labels

def run_clustering_step(
    track_features: pd.DataFrame,
    clustering_method: Callable,
    clustering_step_name: str,
    **kwargs,
) -> pd.Series:
    """
    Runs a clustering step and returns the labels
    """
    X = track_features[relevant_features]
    labels = clustering_method(X, **kwargs)
    return labels


def get_grid_search_params(
    eps_min: float = 0.2,
    eps_max: float = 0.4,
    eps_step: float = 0.1,
    min_samples_min: int = 10,
    min_samples_max: int = 40,
    min_samples_step: int = 10,
):
    grid_search_params = []
    eps = eps_min
    while eps <= eps_max:
        min_samples = min_samples_min
        while min_samples <= min_samples_max:
            grid_search_params.append({
                "eps": eps, "min_samples": min_samples
            })
            min_samples += min_samples_step
        eps = round(eps + eps_step, 1)
    return grid_search_params



def DBSCAN_grid_search(
    track_features: pd.DataFrame,
    grid_search_params: List[Dict[str, int | float]],
    output_folder: Path,
) -> None:
    for params in tqdm.tqdm(grid_search_params):

        eps = params["eps"]
        min_samples = params["min_samples"]

        # print(f"Running DBSCAN with eps={eps} and min_samples={min_samples}")

        clusters_summary_file_name = f"DBSCAN_summary_eps_{eps}_min_samples_{min_samples}.csv"
        clusters_summary_file = output_folder / clusters_summary_file_name
        scatter_matrix_file_name = f"DBSCAN_plot_eps_{eps}_min_samples_{min_samples}.html"
        scatter_matrix_file = output_folder / scatter_matrix_file_name
        if clusters_summary_file.exists() and scatter_matrix_file.exists():
            continue

        X = track_features[relevant_features].to_numpy()
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        track_features["cluster_id"] = labels

        clusters_summary = summarize_clusters(track_features)
        clusters_summary.to_csv(clusters_summary_file, index=False)

        visualize_clusters(track_features, filename=scatter_matrix_file.as_posix())


def run_clustering_pipeline(
    track_features: pd.DataFrame,
) -> pd.DataFrame:
     
    X = track_features[relevant_features].to_numpy()
    labels = DBSCAN(
        eps=0.2,
        min_samples=5,
    ).fit_predict(X)
    track_features["cluster_id"] = labels
    return track_features

def get_cluster_summary(cluster: pd.DataFrame) -> pd.Series:
    summary: dict = dict()
    for feature in relevant_features:
        feature_min = cluster[feature].min()
        feature_max = cluster[feature].max()
        feature_range = feature_max - feature_min
        summary.update({
            f"{feature}_min": feature_min,
            f"{feature}_max": feature_max,
            f"{feature}_range": feature_range,
        })
    summary.update({"num_tracks": len(cluster)})
    return pd.Series(summary)

def summarize_clusters(track_features: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of features for each cluster
    Args:
        track_features: pd.DataFrame
            A table with the tracks and their labels
    Returns:
        clusters_summary: pd.DataFrame
            A table where each row represents a cluster,
            and the columns represent the min/max/range of each relevant feature
    """
    clusters_summary = track_features.groupby(
        by="cluster_id",
        as_index=False,
    ).apply(get_cluster_summary)
    return clusters_summary
    

def visualize_clusters(
    track_features: pd.DataFrame,
    filename: str = "clusters_scatter_matrix.html",
) -> None:
    """
    Plot the scatter matrix of relevant features,
    assigning a unique color to each cluster
    """
    track_features["cluster_id"] = track_features.cluster_id.astype(str)
    fig = px.scatter_matrix(
        data_frame=track_features,
        dimensions=relevant_features,
        color="cluster_id",
        hover_name="track_name",
    )
    plotly.offline.plot(fig, filename=filename, auto_open=False)


if __name__ == "__main__":
    output_folder = Path("DBSCAN_results")
    output_folder.mkdir(exist_ok=True)
    grid_search_params = get_grid_search_params()
    track_features = pd.read_csv("track_features.csv")
    DBSCAN_grid_search(
        track_features=track_features,
        grid_search_params=grid_search_params,
        output_folder=output_folder,
    )
