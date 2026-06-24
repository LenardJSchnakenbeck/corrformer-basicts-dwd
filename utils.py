import pickle
import shutil
from pathlib import Path

import mlflow
from basicts.configs import BasicTSForecastingConfig
from basicts.utils import BasicTSMode
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import logging
import importlib
from basicts.runners.builder import Builder
from easytorch.utils import get_logger, set_visible_devices
from basicts.configs.base_config import BasicTSConfig
from basicts.runners import BasicTSRunner
from easytorch.device import set_device_type
from easytorch.utils import set_visible_devices


def get_inference_func(
        cfg: BasicTSConfig,
        gpus: str | None = None,
        batch_size: int | None = None,
        scaler = None,
):

    logger = get_logger("BasicTS-launcher")
    logger.info("Launching BasicTS evaluation.")

    set_device_type("gpu" if gpus else "cpu")
    if gpus:
        set_visible_devices(gpus)

    cfg.gpus = gpus
    cfg.gpu_num = len(gpus.split(",")) if gpus else 0

    if batch_size is not None:
        cfg.test_batch_size = batch_size

    runner = BasicTSRunner(cfg)
    if scaler is None:
        train_dataset = Builder._build_dataset(cfg, BasicTSMode.TRAIN)
        runner.scaler.fit(train_dataset.data)
    else:
        runner.scaler = scaler

    runner.init_logger(logger_name="BasicTS-inference", log_file_name="inference_log")
    return runner.inference

def get_metrics_from_tfevents(tf_event_path):
    ea = EventAccumulator(str(tf_event_path))
    ea.Reload()

    metrics = {}
    for tag in ea.Tags()['scalars']:
        values = [round(step.value, 3) for step in ea.Scalars(tag)]
        metrics[tag] = values

    return metrics


def build_model(cfg):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return Builder._build_model(cfg, logger)

def update_md5(config):
    config._serialized = config._serialize()
    config._md5 = config._get_md5(config.serialized)
    return config

def get_module(cfg_module):
    """
    Get the module from the configuration module.
    Example:
            module_config = get_module("prediction_Corrformer")
            config = module_config.config

    Args:
        cfg_module (str): The module name.

    Returns:
        module: The module.
    """
    if cfg_module.endswith(".py"):
        cfg_module = cfg_module[:-3]
    cfg_module = importlib.import_module(cfg_module)
    return cfg_module

def resolve_config(config: str|BasicTSForecastingConfig):
    """
    resolves, whether config is the name of a module or a BasicTSForecastingConfig object and returns the config (and module)
    """
    if isinstance(config, str):
        module = get_module(config)
        return module.config, module
    return config, None

def clear_dir(dir_path):
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
        dir_path.mkdir()

def toggle_logging(silence=True):
    loggers = [
        "tqdm",
        "tqdm.cli"
    ]
    for logger in loggers:
        level = logging.WARNING if silence else logging.INFO
        logging.getLogger(logger).setLevel(level)


def set_attributes(obj, path, value):
    """
    Set attributes of a python object using a dot-separated path.

    Args:
        obj: The object to set attributes on.
        path: A dot-separated string of attribute names.
        value: The value to set.
    """
    parts = path.split(".")

    current = obj

    for p in parts[:-1]:
        if isinstance(current, dict):
            current = current[p]
        else:
            current = getattr(current, p)

    last = parts[-1]

    if isinstance(current, dict):
        current[last] = value
    else:
        setattr(current, last, value)


def log_tfevents_dir_to_mlflow(dir_path, pattern="events.out.tfevents.*"):
    event_file_paths = list(dir_path.glob(pattern))
    for event_file_path in event_file_paths:
        print(f"Importing metrics from {event_file_path.name}")
        ea = EventAccumulator(str(event_file_path))
        ea.Reload()

        tags = ea.Tags()['scalars']
        for tag in tags:
            events = ea.Scalars(tag)
            for event in events:
                mlflow.log_metric(
                    key=tag,
                    value=event.value,
                    step=event.step
                )


def log_to_mlflow(experiment_name, local_log_dir, config, save_in_mlflow_db = False):
    """
    Logs experiment setting in mlflow:
    - model weights (if save_in_mlflow_db) or checkpoint directory
    - config
    - scaler (if save_in_mlflow_db) or scaler path
    - tensorboard logs

    Args:
        experiment_name: name of the experiment in mlflow
        local_log_dir: directory of the tensorboard-logs, usually Path(config.ckpt_save_dir) / Path(config_hash)
        config: config of the ai pipeline, something like BasicTSForecastingConfig
        save_in_mlflow_db: whether to save the model weights in mlflow
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    #mlflow.log_artifact(config_module.__file__)

    # log model weights
    if save_in_mlflow_db:
        for pt_file in Path(local_log_dir).rglob("*.pt"):
            mlflow.log_artifact(str(pt_file), artifact_path="model-checkpoints")
    else:
        mlflow.log_param("checkpoint-directory", str(local_log_dir))

    # log config (hyperparameter results)
    config_path = local_log_dir / "config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    mlflow.log_param("config-path", str(config_path))

    # log fitted scaler
    dataset = Builder._build_dataset(config, BasicTSMode.TRAIN)
    scaler = config.scaler
    fitted_scaler = scaler(norm_each_channel=config.norm_each_channel, rescale=config.rescale)
    fitted_scaler.fit(dataset.data)

    scaler_path = local_log_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(fitted_scaler, f)

    mlflow.log_param("scaler-path", str(scaler_path))
    if save_in_mlflow_db:
        mlflow.log_artifact(str(scaler_path), artifact_path="scaler")

    # log tensorboard events
    tensorboard_path = local_log_dir / "tensorboard"
    log_tfevents_dir_to_mlflow(tensorboard_path, "events.out.tfevents.*")

def load_from_mlflow_or_pickle(
    config_path: str | Path | None,
    scaler_path: str | Path | None = None,
    mlflow_run_id: str | None = None,
    experiment_name: str | None = None,
):
    """
    Load config and scaler from either direct pickle paths or from MLflow params.

    Priority: explicit paths > mlflow run > search by experiment name

    Returns:
        config: the deserialized config object
        scaler: a fitted scaler instance (reconstructed from the saved class + config)
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    if mlflow_run_id:
        run = mlflow.get_run(mlflow_run_id)
    elif experiment_name:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"No MLflow experiment found with name: {experiment_name}")
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise ValueError(f"No runs found in experiment: {experiment_name}")
        run = runs[0]
    else:
        raise ValueError("Provide either explicit paths, an mlflow_run_id, or an experiment_name")

    params = run.data.params
    config_path = config_path or params.get("config-path")


    if not config_path:
        raise ValueError("Could not resolve config-path from MLflow run params")

    with open(config_path, "rb") as f:
        config = pickle.load(f)

    if scaler_path is None:
        return config, None
    else:
        scaler_path = scaler_path or params.get("scaler-path")
        if not scaler_path:
            raise ValueError("Could not resolve scaler-path from MLflow run params")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        return config, scaler

class ConfigUpdater:
    def __init__(self, search_space = None, space = None):
        self.search_space = search_space or {}
        self.space = space or {}

    def set_param(self, path: str, arg):
        self.space[path] = arg
        return self

    def set_param_to_search(self, path: str, dtype: type, args: tuple, kwargs: dict | None = None):
        """
        Register a hyperparameter. Path is a dot-separated string starting with an attribute of the config,
        followed by their attribute or index, and so on.

        Example:
            set_param_to_search("optimizer_params.lr", float, ("lr", 1e-5, 1e-2), {"log": True})
        """
        self.search_space[path] = (dtype, args, kwargs or {})
        return self

    def _suggest(self, trial, name, dtype, args, kwargs):
        """
        creates optuna suggestion method.
        """
        if dtype is float:
            return trial.suggest_float(name, *args, **kwargs)
        elif dtype is int:
            return trial.suggest_int(name, *args, **kwargs)
        elif dtype in (str, list, tuple):
            return trial.suggest_categorical(name, args[0])
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def _set_path(self, obj, path, value):
        """
        Set a value in a nested object by a dot-separated path.

        Args:
            obj: the object to set the value in (e.g. a config object)
            path: the dot-separated path of the attribute
            value: the value to set
        """
        parts = path.split(".")
        current = obj

        for p in parts[:-1]:
            if isinstance(current, dict) or isinstance(current, (list, tuple)) and isinstance(p, int):
                current = current[p]
            else:
                current = getattr(current, p)

        last = parts[-1]
        if isinstance(current, dict) or isinstance(current, (list, tuple)) and isinstance(p, int):
            current[last] = value
        else:
            setattr(current, last, value)

    def update(self, config, trial):
        if self.search_space:
            for path, (dtype, args, kwargs) in self.search_space.items():
                name = args[0]
                value = self._suggest(trial, name, dtype, args, kwargs)
                self._set_path(config, path, value)

        if self.space:
            for path, value in self.space.items():
                self._set_path(config, path, value)

        config = update_md5(config)
        return config