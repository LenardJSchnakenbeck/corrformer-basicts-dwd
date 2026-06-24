import pytest
import logging
from pathlib import Path
import torch
import numpy as np
from basicts.launcher import BasicTSLauncher
from utils import clear_dir, get_inference_func, get_metrics_from_tfevents, resolve_config, toggle_logging
from prediction_Corrformer import Dwd_Temp_Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compare_shapes(actual_shape, expected_shape, context:str = "no context"):
    assert len(actual_shape) == len(expected_shape), (
        f"Shape mismatch! Actual: {actual_shape}, Expected: {expected_shape}"
    )

    shape_encoding = {
        "T": "Timesteps",
        "num_features": "Features",
        "B": "Batch size",
    }
    for dataset_dim, expected_dim in zip(actual_shape, expected_shape):
        if isinstance(expected_dim, int):
            assert dataset_dim == expected_dim, f"Expected {expected_dim}, got {dataset_dim} (context:{context})"
        elif expected_dim in shape_encoding:
            logger.info(f"{dataset_dim}, {shape_encoding[expected_dim]}")
        else:
            raise ValueError(f"Invalid expected dimension: {expected_dim} (out of {shape_encoding})")


class DatasetTests:
    def __init__(self,
                 config,
                 dataset,
                 expected_shape=(1, "T", "num_features"),
                 ):
        logger.info(f"Testing dataset: {dataset}")
        self.config = config
        self.dataset = dataset
        self.expected_shape = expected_shape
        self.data = self.dataset.data

    def test_shapes(self):
        compare_shapes(self.data.shape, self.expected_shape, context="Dataset output shapes")

    def test_no_nans(self):
        x = self.data
        num_nans = torch.isnan(x).sum()
        assert num_nans == 0, f"Dataset contains {num_nans} NaNs"

    def test_no_infs(self):
        x = self.data
        num_infs = torch.isinf(x).sum()
        assert num_infs == 0, f"Dataset contains {num_infs} Infs"

    def test_timestamps_matrix(self, timestamps_matrix):
        diffs = np.diff(timestamps_matrix)

        assert np.all(diffs > 0), \
            "Timestamps are not strictly increasing"

        assert np.all(diffs == 1), \
            "Timestamps are not hourly spaced"

    def conduct_tests(self):
        self.test_shapes()
        self.test_no_nans()
        self.test_no_infs()


class Dataslice(Dwd_Temp_Dataset):
    def __init__(self, input_len, pred_len, label_len, dataset_name="Dataslice", mode="train", memmap=False, length = 10):
        super().__init__(
            dataset_name = dataset_name,
            input_len = input_len,
            pred_len = pred_len,
            label_len = label_len,
            mode = "train",
            memmap = memmap,
        )
        self.length = length
        import copy
        parent_getitem = super().__getitem__
        self.items = [copy.deepcopy(parent_getitem(i)) for i in range(length)]

    def __getitem__(self, index: int) -> dict:
        return self.items[index]

    def __len__(self) -> int:
        return self.length

    @property
    def data(self):
        tensor_list = [
            torch.cat((item["inputs"], item["targets"]), dim=1)
            for item in self.items
        ]
        return torch.cat(tensor_list, dim=1)

class ModelTest:
    def __init__(self,
                 dataset,
                 expected_input_shape,
                 expected_output_shape,
                 config
                 ):
        self.dataset = dataset
        self.expected_input_shape = expected_input_shape
        self.expected_output_shape = expected_output_shape
        self.data = self.dataset.data
        self.config = config
        self.test_run_metrics = []
        self.smoke_test_num_epochs = 5
        self.overfitting_test_num_epochs = 30
        self.inference_func = None

    def test_model(self):
        if self.inference_func is None:
            self.inference_func = get_inference_func(self.config, self.config.gpus)

        prediction = self.inference_func(self.dataset[0])
        compare_shapes(prediction.shape, self.expected_output_shape, context="Model output shapes")

    @staticmethod
    def test_metric(metric: list):
        assert np.all(np.isfinite(metric)), f"Loss became infinite"
        assert not np.any(np.isnan(metric)), f"Loss became NaN"

    def rename_checkpoint(self):
        """fixes a bug, where the checkpoint path is dependent on the number of epochs"""
        old_path = self.experiment_dir / (self.config.model.__name__ + f"_{self.smoke_test_num_epochs}.pt")
        new_path = self.experiment_dir / (self.config.model.__name__ + f"_0{self.smoke_test_num_epochs}.pt")
        if old_path.exists():
            old_path.rename(new_path)

    def smoke_test(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.smoke_test_num_epochs
        self.config.num_epochs = num_epochs
        logger.info(f"smoke test with {num_epochs} epochs")
        toggle_logging(silence=True)
        BasicTSLauncher.launch_training(self.config)
        toggle_logging(silence=False)


        if num_epochs > 1:
            tfevents_paths = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
            assert len(tfevents_paths) == 1, "more than one log file for smoke test found (just the first one is used)."
            metrics = get_metrics_from_tfevents(str(tfevents_paths[0]))
            self.test_metric(metrics["train/loss"])
            assert np.all(np.diff(metrics["train/loss"]) < 0), "training loss is not strictly decreasing"
            assert metrics["train/loss"][-1] <= metrics["train/loss"][0], "epoch 1 has a smaller loss than last epoch"

    #@pytest.mark.slow
    def overfitting_test(self, num_epochs=None):
        if num_epochs == None:
            num_epochs = self.overfitting_test_num_epochs

        self.config.num_epochs = num_epochs
        logger.info(f"overfitting test with {num_epochs} epochs")
        toggle_logging(silence=True)
        BasicTSLauncher.launch_training(self.config)
        toggle_logging(silence=False)

        tfevents_paths = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
        metrics = get_metrics_from_tfevents(str(tfevents_paths[0]))
        self.test_metric(metrics["train/loss"])

    def create_dataslice(self, num_items=2):
        dataset_name = self.config.dataset_name
        self.config.dataset_name = dataset_name
        self.config.dataset_type = Dataslice
        self.config.dataset_params = {
            "input_len": self.config.model_config.seq_len,
            "pred_len": self.config.model_config.pred_len,
            "label_len": self.config.model_config.label_len if hasattr(self.config.model_config, "label_len") else None,
            "dataset_name": dataset_name,
            "mode": "train",
            "memmap": False
        }

        self.experiment_dir = Path(self.config.ckpt_save_dir) / Path(self.config.md5)
        self.tensorboard_dir = self.experiment_dir / "tensorboard"

    def conduct_tests(self):
        self.create_dataslice()
        clear_dir(self.experiment_dir)
        self.test_model()
        self.smoke_test(None)
        self.rename_checkpoint()
        self.overfitting_test()

def test_ai_pipeline(test_ai_pipeline_params):
    config, expected_input_shape, expected_output_shape = test_ai_pipeline_params
    config, module_config = resolve_config(config)
    DatasetClass = config.dataset_type
    dataset = DatasetClass(**config.dataset_params)

    DatasetTester = DatasetTests(dataset=dataset, config=config)
    DatasetTester.conduct_tests()

    ModelTester = ModelTest(
        dataset=dataset,
        expected_input_shape=expected_input_shape,
        expected_output_shape=expected_output_shape,
        config=config
    )
    ModelTester.conduct_tests()


if __name__ == "__main__":
    #test_ai_pipeline_params = ("prediction_DLinear_custom", (1, "T", "num_features"), (1, "T", "num_features"))
    test_ai_pipeline_params = ("prediction_Corrformer", (1, "T", "num_features"), (1, 24, 480))
    test_ai_pipeline(test_ai_pipeline_params)