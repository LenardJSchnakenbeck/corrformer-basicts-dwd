from typing import Union
import numpy as np
from basicts.data.base_dataset import BasicTSDataset
from basicts.utils.constants import BasicTSMode
import os
import torch

class Dwd_Temp_Dataset(BasicTSDataset):
    """Custom dataset for your specific data format."""

    def __init__(
            self,
            dataset_name: str,
            input_len: int,
            pred_len: int,
            label_len: int,
            mode: BasicTSMode | str,
            timestamps_features: list[str] | None = ('hourofday', 'dayofweek', 'dayofmonth', 'dayofyear'),
            timestamps_freq: int = 60,
            repeat_timestamps: bool = False, #Whether to create a timestamp for each station in a batch (they all have the same timestamp), or just use one for the whole batch (as for Corrformer)
            data_file_path: str | None = None,
            handling_nan: str | None = "impute", # "impute", "mask"
            memmap: bool = False
    ) -> None:

        """
        Initializes the BasicTSForecastingDataset by setting up paths, loading data, and
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            input_len (int): The length of the input sequence (number of historical points).
            pred_len (int): The length of the output sequence (number of future points to predict).
            mode (Union[BasicTSMode, str]): The mode of the dataset, indicating whether it is for training, validation, or testing.
            repeat_timestamps (bool): Whether to create a timestamp for each station in a batch (they all have the same timestamp), or just use one for the whole batch (as for Corrformer)
            local (bool): Flag to determine if the dataset is local.
            data_file_path (str | None): Path to the file containing the time series data. Default to "datasets/{name}".
            memmap (bool): Flag to determine if the dataset should be loaded using memory mapping.
        """
        super().__init__(dataset_name, mode, memmap)
        if data_file_path is None:
            data_file_path = f"datasets/{dataset_name}"  # default file path

        try:
            self._data = np.load(
                os.path.join(data_file_path, f"{mode}_data.npy"),
                mmap_mode="r" if memmap else None)

            self.spatial_data = np.load(
                os.path.join(data_file_path, f"spatial_data.npy"),
                mmap_mode="r" if memmap else None)

            if timestamps_features:
                self.timestamps = np.load(
                    os.path.join(data_file_path, f"{mode}_timestamps.npy"),
                    mmap_mode="r" if memmap else None)
                # timestamps encoding in dwd_download.py>create_timestamps()
                timestamp_encoding = {
                    "hourofday": 0,
                    "dayofweek": 1,
                    "dayofyear": 2,
                    "month": 3,
                    "year": 4,
                    "dayofmonth": 5,
                }
                timestamps_features = [timestamp_encoding[t_feature] for t_feature in timestamps_features]
                self.timestamps = self.timestamps[:, :, timestamps_features]


        except FileNotFoundError as e:
            raise FileNotFoundError(f"Cannot load dataset from {data_file_path}, Please set a correct local path." 
                                    "If you want to download the dataset, please set the argument `local` to False.") from e

        self._data = torch.tensor(self._data)
        if handling_nan == "mask":
            self.mask = self.masking(self._data)
        elif handling_nan == "impute":
            self._data = self.impute_knn(self._data)

        self.spatial_data = torch.tensor(self.spatial_data, dtype=torch.float32)
        self.timestamps = torch.tensor(self.timestamps)
        self.timestamps_freq = timestamps_freq

        self.input_len = input_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.memmap = memmap
        self.timestamps_features = timestamps_features
        self.repeat_timestamps = repeat_timestamps

    @staticmethod
    def masking(data):
        mask = ~torch.isnan(data)
        data = torch.nan_to_num(data, nan=0.0)
        return data, mask

    @staticmethod
    def impute_knn(tensor: torch.Tensor, n_neighbors: int = 5) -> torch.Tensor:
        from sklearn.impute import KNNImputer
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.requires_grad:
            tensor = tensor.detach()

        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        numpy_data = tensor.squeeze(0).numpy()
        imputed_data = imputer.fit_transform(numpy_data)

        return torch.tensor(imputed_data, dtype=tensor.dtype).unsqueeze(0)

    @staticmethod
    def generate_future_timestamps(last_dt, pred_len, freq):
        future = []

        for i in range(1, pred_len + 1):
            dt = last_dt + i * freq

            future.append([
                dt.hour,
                dt.minute,
                dt.timetuple().tm_yday,
                dt.weekday(),
                dt.year
            ])
        return torch.tensor(future, dtype=torch.float32).unsqueeze(0)

    def create_item(self, history_data, future_data, history_timestamps = None, future_timestamps = None):
        """
        Creates a dictionary containing the input and target data for a single time series.

        Args:
            history_data (torch.Tensor): The historical data.
            future_data (torch.Tensor): The future data. Not given for inference.
            history_timestamps (torch.Tensor, optional): The historical timestamps. Defaults to None.
            future_timestamps (torch.Tensor, optional): The future timestamps. Defaults to None.
        """
        item = {}
        item["inputs"] = history_data.clone() if self.memmap else history_data
        item["spatial_inputs"] = self.spatial_data.clone() if self.memmap else self.spatial_data
        if future_data is not None:
            item["targets"] = future_data.clone() if self.memmap else future_data
        item["dec_inputs"] = torch.cat([
            item["inputs"][:,-self.label_len:, :].clone()
            if self.memmap else item["inputs"][:,-self.label_len:, :],
            # torch.zeros_like(item["targets"])
            torch.zeros(
                item["inputs"].shape[0], # batch size = 1
                self.pred_len,
                item["inputs"].shape[2], # num_stations
                # device=future_data.device
            )
        ], dim=1)

        if self.timestamps_features:
            item["inputs_timestamps"] = history_timestamps.clone() if self.memmap else history_timestamps
            if future_timestamps is not None:
                item["targets_timestamps"] = future_timestamps.clone() if self.memmap else future_timestamps
                item["dec_inputs_timestamps"] = torch.cat([
                    item["inputs_timestamps"][:, -self.label_len:, :].clone()
                    if self.memmap else item["inputs_timestamps"][:, -self.label_len:, :],
                    item["targets_timestamps"]
                ], dim=1)
            else:
                # inference
                item["dec_inputs_timestamps"] = torch.cat([
                    item["inputs_timestamps"][:, -self.label_len:, :].clone()
                    if self.memmap else item["inputs_timestamps"][:, -self.label_len:, :],
                    self.generate_future_timestamps(
                        item["inputs_timestamps"][0, -1],
                        self.pred_len,
                        self.timestamps_freq
                    )
                ], dim=1)

            num_stations = item["inputs"].shape[-1]
            timestamp_keys = [key for key in item.keys() if "timestamp" in key]
            for key in timestamp_keys:
                if self.repeat_timestamps:
                    item[key] = item[key].repeat(num_stations, 1, 1)
                else:
                    item[key] = item[key]
        """
        for key in item.keys():
            if "spatial" not in key:
                item[key] = item[key].squeeze(0)
        """
        return item

    def __getitem__(self, index: int, num_spatial_factors: int = 3) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing "inputs" and "targets", where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """

        history_data = self._data[:, index: index + self.input_len]
        future_data = self._data[:, index + self.input_len: index + self.input_len + self.pred_len]

        if self.timestamps_features:
            history_timestamps = self.timestamps[:, index: index + self.input_len, :]
            future_timestamps = self.timestamps[:, index + self.input_len: index + self.input_len + self.pred_len, :]
        else:
            history_timestamps = None
            future_timestamps = None

        return self.create_item(history_data, future_data, history_timestamps, future_timestamps)

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        #return len(self._data[1]) - self.input_len - self.pred_len + 1
        _, T, _ = self._data.shape
        return T - self.input_len - self.pred_len + 1

    @property
    def data(self) -> np.ndarray:
        return self._data

if __name__ == "__main__":
    import json

    input_len = 48
    pred_len = 24
    label_len = input_len // 2
    dataset_name = "dwd_weather"
    dataset_meta = json.load(open(f"datasets/{dataset_name}/meta.json"))

    dataset = Dwd_Temp_Dataset(
        dataset_name=dataset_name,
        input_len=input_len,
        pred_len=pred_len,
        label_len=label_len,
        mode="train",
        timestamps_features=['hourofday', 'dayofweek', 'dayofyear', 'month', 'year'],
        repeat_timestamps=False,
        data_file_path=f"datasets/{dataset_name}",
        handling_nan="impute",
        memmap=False
    )

    print(len(dataset))
    item = dataset[0]


