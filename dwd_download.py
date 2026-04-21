import pandas as pd
import polars as pl
from wetterdienst.provider.dwd.observation import DwdObservationRequest
from dataclasses import dataclass, field
import numpy as np
import math
from pathlib import Path
import json
import torch

## FETCH DATA

def fetch_dwd_data(cfg) -> pd.DataFrame:
    """
    it is possible to fetch data from different datasets, e.g. climate_summary and solar,
    and they may deliver different values for the same conditions, which throws an error when pivoting the df.
    """

    station_filter = cfg.station_filter
    measurements = cfg.measurements
    dwd_query = cfg.dwd_query

    request = DwdObservationRequest(**dwd_query)
    if isinstance(station_filter, dict):
        stations = request.filter_by_distance(**station_filter)
    elif isinstance(station_filter, list):
        stations = request.filter_by_station_id(station_filter)
    else:
        stations = request.all()
    stations = stations.values.all()
    df = stations.df

    df_wide = df.pivot(
        values="value",
        index=["station_id", "date"],
        on="parameter"
    )
    if measurements:
        df_wide = df_wide.select(["station_id", "date"] + measurements)

    print(f"Data fetched from DWD: {df_wide.shape[0]} lines & {df_wide.shape[1]} columns")

    return df_wide.to_pandas()

def fetch_stations_coords(cfg) -> pd.DataFrame:
    """
    Get all stations and their metadata.
    """

    request = DwdObservationRequest(**cfg.dwd_query)
    stations = request.all()
    stations_df = stations.df
    stations_df = stations_df[['station_id', 'latitude', 'longitude', 'height']].rename({"height": "altitude"}).to_pandas()
    return stations_df.sort_values(['station_id']) #.astype(float) # set_index("station_id").sort_index()

## IO UTILS

def load_csv_from_disk(path: str | Path):
    return pd.read_csv(path)

def save_csv_to_disk(df: pd.DataFrame | pl.DataFrame, path: str):
    path_dir = Path(path).parent
    path_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(df, pd.DataFrame):
        df.to_csv(path, index=False)
    elif type(df) == pl.DataFrame:
        df.fill_nan(math.nan).write_csv(path)

## PREPROCESSING

def set_dtypes_date_stationid(df):
    df.date = pd.to_datetime(df.date)
    df['date'] = df['date'].dt.as_unit('us')
    df.station_id = df['station_id'].astype(int)
    return df

def stations_data_completeness(df):
    """
    creates df, that shows data completeness for each station, to determine top_x_stations manually
    """
    station_stats = df.groupby('station_id')['temperature_air_mean_2m'].count().to_frame('count')
    station_stats['percentage'] = (station_stats['count'] / station_stats[
        'count'].max()) * 100  # percentage relative to station with most values
    return station_stats.sort_values(by='count', ascending=False).reset_index(names=["station_id"])

def preprocess_weather_data(raw_df: pd.DataFrame, top_x_stations: int) -> pd.DataFrame:
    def drop_na_columns(df):
        threshold = len(df) * (2 / 3)
        df_clean = df.dropna(axis=1, thresh=threshold)
        return df_clean

    def select_top_x_stations(df, top_x_stations):
        station_stats = stations_data_completeness(df)
        top_x_ids = station_stats["station_id"].head(top_x_stations)
        return df[df['station_id'].isin(top_x_ids)]

    def parse_date(df):
        df['date'] = pd.to_datetime(df['date'])
        return df

    df = parse_date(raw_df)
    #df_clean = drop_na_columns(raw_df)
    if top_x_stations:
        df = select_top_x_stations(df, top_x_stations)
    print(f"{top_x_stations} stations out of {raw_df['station_id'].nunique()} with fewest nan values are picked")

    return df

## FEATURES

def get_spatial_data(cfg, station_ids):
    spatial_data = fetch_stations_coords(cfg)
    spatial_data["station_id"] = spatial_data["station_id"].astype(int)
    spatial_data = spatial_data[
        spatial_data["station_id"].isin(station_ids)
    ]
    return torch.tensor(
        spatial_data[["latitude", "longitude", "altitude"]].values
    )

def stations_df_to_tensor(df, value_col="temperature_air_mean_2m"):
    """
    df: pd.DataFrame with multiindex ["date", "station_id"]
    value_col: column in df with feature

    return: torch.tensor of shape [num_time_steps, num_features (1), num_spatial]
    """

    wide = (
        df[value_col]
        .unstack("station_id")  # -> [date, station]
        .sort_index()
    )

    return torch.tensor(wide.to_numpy(), dtype=torch.float32).unsqueeze(0)

def create_timestamps(date):
    """
    df: pd.DataFrame with multiindex ["date", "station_id"]

    return: torch.tensor of shape [1, num_time_steps, num_timestamps (5)]
    """
    # timestamps are selected using the dataset config
    timestamps = pd.DataFrame({
        "hourofday": date.dt.hour,
        "dayofweek": date.dt.dayofweek,
        "dayofyear": date.dt.dayofyear,
        "month": date.dt.month,
        "year": date.dt.year,
        "dayofmonth": date.dt.day,
    })
    # shape: [1, num_time_steps, num_timestamps (6)]
    return torch.tensor(timestamps.to_numpy(), dtype=torch.float32).unsqueeze(0)

## DATASET FORMAT

def split_data(data, time_stamps, train_val_test_ratio):
    """
    Split data into train, val, and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Input data
    time_stamps : pd.DataFrame
        Timestamps for the input data
    train_val_test_ratio : list[float], optional
        Split ratio for train, val, and test sets (default is (0.5, 0.1, 0.4))

    Returns
    -------
    data_split : dict
        Dictionary containing the split data and timestamps
    """

    if data.shape[1] != time_stamps.shape[1]:
        raise ValueError(f"data has {data.shape[1]} examples, but there are {time_stamps.shape[1]} timestamps")

    num_samples = data.shape[1]
    train_end = int(train_val_test_ratio[0] * num_samples)
    val_end = int((train_val_test_ratio[0] + train_val_test_ratio[1]) * num_samples)

    data_split = {}
    # data shape: batch_size(1),

    data_split["train_data"] = data[:, :train_end, :]
    data_split["train_timestamps"] = time_stamps[:, :train_end, :]

    data_split["val_data"] = data[:, train_end:val_end, :]
    data_split["val_timestamps"] = time_stamps[:, train_end:val_end, :]

    data_split["test_data"] = data[:, val_end:, :]
    data_split["test_timestamps"] = time_stamps[:, val_end:, :]
    return data_split

def write_npys(
        path_dir: Path,
        data_split: dict,
        spatial_data,
        splits: list[str] = ["train", "val", "test"],
        array_types: list[str] = ["data", "timestamps"],
        ):
    """
    Write data and timestamps to .npy files.
    Every combination of split and array type should exist as key in data_split

    Parameters
    ----------
    data_split : dict
        Dictionary containing the split data and timestamps
    splits : list[str], optional
        List of subset names (default is ["train", "val", "test"])
    array_types : list[str], optional
        List of array types (default is ["data", "timestamps"])

    Returns
    -------
    None
    """
    for subset in splits:
        for array_type in array_types:
            key = f"{subset}_{array_type}"
            array = np.asarray(data_split[key], dtype=np.float32)
            path_file = path_dir / f"{key}.npy"
            path_file.unlink(missing_ok=True)
            np.save(path_file, array)

    if spatial_data is not None:
        path_spatial = Path(path_dir) / "spatial_data.npy"
        print(f"spatial_data is saved at {path_spatial}")
        path_spatial.unlink(missing_ok=True)
        np.save(path_spatial, spatial_data)

def write_meta_json(data, temporal_data, spatial_data, train_val_test_ratio, path_dir):
    meta = {
        "name": "dwd_weather",
        "domain": "weather",
        "frequency (minutes)": 60,
        "shape": list(data.shape),
        "timestamps_shape": list(temporal_data.shape),
        "spatial_shape": list(spatial_data.shape),
        "timestamps_description": [
            "hour of day",
            "day of week",
            "day of year",
            "month of year",
            "year",
            "day of month"
        ],
        "num_time_steps": temporal_data.shape[1],
        "num_vars": data.shape[0],
        "has_graph": False,
        "regular_settings": {
            "train_val_test_ratio": train_val_test_ratio,
            "norm_each_channel": True,
            "rescale": False,
            "metrics": ["MAE", "MSE"],
            "null_val": math.nan
        }
    }

    path_file = Path(path_dir) / "meta.json"
    path_file.unlink(missing_ok=True)
    with open(path_file, "w") as f:
        json.dump(meta, f, indent=4)

def save_as_basicts_dataset(data, timestamps, spatial_data, path_dir, train_val_test_ratio: list[float] = (0.5, 0.1, 0.4)):
    ARRAY_TYPES = ["data", "timestamps"]
    SPLITS = ["train", "val", "test"]

    # ensure directory exists
    path_dir = Path(path_dir)
    path_dir.mkdir(parents=True, exist_ok=True)

    data_split = split_data(data, timestamps, train_val_test_ratio=train_val_test_ratio)
    write_npys(path_dir, data_split, spatial_data, SPLITS, ARRAY_TYPES)
    write_meta_json(data, timestamps, spatial_data, train_val_test_ratio=train_val_test_ratio, path_dir=path_dir)


def build_weather_dataset(cfg):
    csv_path = Path(cfg.csv_path)
    dataset_dir = Path(cfg.dataset_dir)
    overwrite = cfg.overwrite
    top_x_stations = cfg.top_x_stations
    if Path(csv_path).exists() and not overwrite:
        print("Data is loaded from disk")
        weather_df = load_csv_from_disk(csv_path)
    else:
        print(f"Downloading and converting weather data for {dataset_dir}...")
        weather_df = fetch_dwd_data(cfg)
        print("Data fetched successfully!")
    weather_df = set_dtypes_date_stationid(weather_df)
    save_csv_to_disk(weather_df, csv_path)
    print(f"Data saved at {csv_path}")

    weather_df = preprocess_weather_data(weather_df, top_x_stations=top_x_stations)
    print("Data is preprocessed")
    weather_df = weather_df[["date", "station_id"] + cfg.measurements]
    weather_df = weather_df.set_index(["date", "station_id"]).sort_index()
    dates = weather_df.index.get_level_values("date").unique().to_series()
    station_ids = weather_df.index.get_level_values("station_id").unique().to_series()

    data = stations_df_to_tensor(weather_df)
    print(f"data has {torch.isnan(data).sum()} nan values out of {data.numel()} total values")
    timestamps = create_timestamps(dates)
    spatial_data = get_spatial_data(cfg, station_ids)
    save_as_basicts_dataset(data, timestamps, spatial_data, dataset_dir)
    print(f"Data saved at {dataset_dir}")
    return weather_df

@dataclass
class FetchDWDWeatherConfig:
    dwd_query: dict = field(
        metadata={"help": "Query configuration used to fetch DWD data."})
    station_filter: dict | (list | tuple) | None = field(
        metadata={"help": "Optional filter specifying which stations to include."})
    measurements: list[str] | tuple[str] | None = field(
        default=("temperature_air_mean_2m",),
        metadata={"help": "List of measurements of each station to include in the dataset."})
    csv_path: str = field(
        default="datasets/dwd_weather/dwd_weather.csv",
        metadata={"help": "Path to CSV file to save and reload processed DWD weather data."})
    dataset_dir: str = field(
        default="datasets/dwd_weather",
        metadata={"help": "Directory where dataset files will be stored."})
    overwrite: bool = field(
        default=True,
        metadata={"help": "Whether to overwrite (or load) existing dataset files."})
    top_x_stations: int = field(
        default=None,
        metadata={"help": "Keep the top X stations with the most complete time-series data."}
    )

if __name__ == "__main__":
    dwd_query = {
        #"start_date": "2024-01-01",
        "start_date": "2025-11-30",
        "end_date": "2025-12-31",
        "parameters": ["hourly", "TEMPERATURE_AIR"],
    }

    station_filter = {
        "latlon": (51.433, 6.765),
        "distance": 50,
        "unit": "km"
    }
    #station_filter = None
    path = "datasets/dwd_weather_test"
    top_x_stations = 480
    ## select top_x_stations:
    # station_stats = stations_data_completeness(df)
    # print(station_stats[station_stats["percentage"] >= 75].tail(10))
    # top_x_stations = int(input("input number to set top_x_stations:"))


    config = FetchDWDWeatherConfig(
        dwd_query = dwd_query,
        station_filter = station_filter,
        csv_path = f"{path}/dwd_weather.csv",
        dataset_dir = path,
        overwrite = True, #overwrite existing data, otherwise local data will be loaded
        measurements = ["temperature_air_mean_2m"],
        top_x_stations = top_x_stations,
    )

    build_weather_dataset(config)
    weather_df = load_csv_from_disk(config.csv_path)

