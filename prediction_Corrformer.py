import json
from basicts.configs import BasicTSForecastingConfig
from basicts import BasicTSLauncher
from dwd_dataset import Dwd_Temp_Dataset
from Corrformer import Corrformer, CorrformerConfig
from dwd_download import FetchDWDWeatherConfig, build_weather_dataset


def configure_pipeline():
    input_len = 48
    pred_len = 24
    label_len = input_len // 2
    timestamp_features = ['hourofday', 'dayofweek', 'dayofmonth', 'dayofyear'] # ['hourofday', 'dayofweek', 'dayofyear', 'month', 'year']
    data_collate_fn = lambda x: x[0]

    dataset_name = "dwd_weather"
    dataset_meta = json.load(open(f"datasets/{dataset_name}/meta.json"))

    num_stations = dataset_meta["shape"][2]
    timestamps_freq = dataset_meta["frequency (minutes)"]
    num_spatial_features = dataset_meta["spatial_shape"][1]

    model_config = CorrformerConfig(
        seq_len = input_len,
        label_len = label_len,
        pred_len = pred_len,
        e_layers = 2,
        d_layers = 1,
        factor = 1,
        factor_temporal = 1,
        factor_spatial = 1,
        enc_tcn_layers = 1,
        dec_tcn_layers = 1,
        enc_in = 10,
        dec_in = 10,
        c_out = 10,
        node_num = 48,
        node_list = [8, 6],
        num_spatial = num_stations,
        num_spatial_features= num_spatial_features,
        timestamp_features=timestamp_features,
        d_model = 768,
        n_heads = 16,
    )

    if model_config.enc_in * model_config.node_num != num_stations:
        raise ValueError(f"enc_in ({model_config.enc_in}) * node_num ({model_config.node_num}) isn't equal to number of stations ({num_stations}), which causes errors in tensor multiplication.")

    dataset_params = {
        "dataset_name": dataset_name,
        "input_len": input_len,
        "pred_len": pred_len,
        "label_len": label_len,
        "mode": "train",
        "timestamps_features": timestamp_features,
        "timestamps_freq": timestamps_freq,
        "repeat_timestamps": False,
        "data_file_path": f"datasets/{dataset_name}",
        "handling_nan": "impute",
        "memmap": False
    }

    config = BasicTSForecastingConfig(
        model = Corrformer,
        model_config = model_config,
        dataset_type = Dwd_Temp_Dataset,
        dataset_name = dataset_name,
        dataset_params = dataset_params,
        seed = 42,
        gpus = "0",
        num_epochs = 100,
        batch_size = 1, # doesn't matter, input has shape [num_stations, num_time_steps, num_features] instead of batches
        train_data_collate_fn = data_collate_fn, # prevent adding a batch dimension
        val_data_collate_fn = data_collate_fn,
        test_data_collate_fn = data_collate_fn,
        memmap = False,
        # optimizer = torch.optim.Adam # default
        optimizer_params = {"lr": 0.0001},
        lr = 0.0001,
        loss = "MSE",
        rescale=True,
    )
    return config, dataset_params

def train(cfg):
    BasicTSLauncher.launch_training(cfg)

def download_data():
    path = "datasets/dwd_weather"
    top_x_stations = 480
    ## select top_x_stations to fulfill this requirement: num_spatial == enc_in * num_node
    # station_stats = stations_data_completeness(df)
    # print(station_stats[station_stats["percentage"] >= 75].tail(10))
    # top_x_stations = int(input("input number to set top_x_stations:"))

    config = FetchDWDWeatherConfig(
        dwd_query = {
        "start_date": "2024-01-01",
        #"start_date": "2025-11-30",
        "end_date": "2025-12-31",
        "parameters": ["hourly", "TEMPERATURE_AIR"],
        },
        station_filter = None,
        csv_path = f"{path}/dwd_weather.csv",
        dataset_dir = path,
        overwrite = True, #False
        measurements = ["temperature_air_mean_2m"],
        top_x_stations = top_x_stations,
    )
    build_weather_dataset(config)


if __name__ == "__main__":
    download_data()
    config, dataset_params = configure_pipeline()
    train(config)


# tensorboard --logdir checkpoints

