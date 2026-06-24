import sys
from basicts import BasicTSLauncher
import mlflow
from pathlib import Path
import argparse
import pytest
from basicts.configs import BasicTSForecastingConfig
from utils import resolve_config, get_metrics_from_tfevents, log_to_mlflow, ConfigUpdater, get_module
import optuna
from copy import deepcopy
import json

def get_args():
    if len(sys.argv) == 1:
        return None

    parser = argparse.ArgumentParser()

    parser.add_argument('--module_config', type=str, required=True,
                        help='name of the module, where the config is defined')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='experiment name in mlflow')
    parser.add_argument('--task', type=str, required=True,
                        help='task to run', choices=['test', 'train', 'tune'])
    parser.add_argument('--config_changes', type=json.loads, required=False,
                        help='config changes to be applied',
                        default='{"num_epochs": 1}')
    parser.add_argument('--hyperparameters', type=json.loads, required=False,
                        help='hyperparameters to be optimized',
                        default='hyperparameters = {"optimizer_params.lr": (float, (1e-5, 1e-2), {"log": True}),'\
                        '"optimizer_params.weight_decay": (float, (1e-6, 1e-2), {"log": True}),}')

    return parser.parse_args()


def run_experiment(config, mlflow_experiment_name):
    config, module_config = resolve_config(config)
    config_hash = config.md5

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run(run_name=config_hash):
        try:
            print("Training started. Press Ctrl+C to stop manually.")
            BasicTSLauncher.launch_training(config)
            mlflow.set_tag("status", "completed")

        except KeyboardInterrupt:
            print("\n[Manual Termination] Ctrl+C. Wrapping up...")
            mlflow.set_tag("status", "manually_stopped")

        except Exception as e:
            print(f"\n[Error] Training crashed: {e}")
            mlflow.set_tag("status", "failed")
            raise e # Re-raise if you want the console to show the full traceback

        finally:
            print("saving Model to MLflow")
            local_log_dir = Path(config.ckpt_save_dir) / Path(config_hash)
            log_to_mlflow(mlflow_experiment_name, local_log_dir, config, save_in_mlflow_db=False)
            print("Model saved to MLflow.")


def run_tests(config_module_name: str|BasicTSForecastingConfig, expected_input_shape, expected_output_shape):
    class ConfigPlugin:
        def __init__(self, config, expected_input_shape, expected_output_shape):
            self.config = config
            self.expected_input_shape = expected_input_shape
            self.expected_output_shape = expected_output_shape

        @pytest.fixture
        def test_ai_pipeline_params(self):
            return self.config, self.expected_input_shape, self.expected_output_shape

    plugin = ConfigPlugin(config_module_name, expected_input_shape, expected_output_shape)

    result = pytest.main(
        [
            "test_ai_pipeline.py",
            "-vv",  # verbose
            #"--log-cli-level=INFO",  # Ensure your pipeline's own logs print alongside the crash
            "-W",
            "ignore::FutureWarning",
            "-W",
            "ignore::pytest.PytestAssertRewriteWarning", # caused by anyio import, despite anyio is importet by pytest
            ],
        plugins=[plugin],
    )
    return result


def hyperparametersearch(config, experiment_name, hyperparameters: dict, config_changes: dict = {"num_epochs": 20}):
    """
    Perform hyperparameter search for a given experiment.

    Args:
        config: config object, or module name, where module.config is the config object.
        experiment_name: experiment_name logged in mlflow.
        config_changes: A dictionary of hyperparameter changes.

    Returns:
        None
    """
    config, module_config = resolve_config(config)

    def objective(trial):
        cfg = deepcopy(config)

        updater = ConfigUpdater(search_space=hyperparameters, space=config_changes)
        cfg = updater.update(cfg, trial)

        run_experiment(cfg, experiment_name)

        local_log_dir = Path(cfg.ckpt_save_dir) / Path(cfg.md5)
        tensorboard_path = local_log_dir / "tensorboard"
        metrics = get_metrics_from_tfevents(tensorboard_path)
        return min(metrics[f"val/{cfg.loss}"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, timeout=None)

    print("Best params:", study.best_params)
    print("Best value: ", study.best_value)


if __name__ == "__main__":
    if args := get_args():
        module_name = args.module_config
        exp_name = args.exp_name
        task = args.task
    else:
        exp_name = "weather_prediction"
        module_name = "prediction_Corrformer"
        task = "tune"

    hyperparameters = {
        "optimizer_params.lr": (float, (1e-5, 1e-2), {"log": True}),
        "optimizer_params.weight_decay": (float, (1e-6, 1e-2), {"log": True}),
    }

    if task == "test":
        module = get_module(module_name)
        run_tests(module_name, module.expected_input_shape, module.expected_output_shape)
    elif task == "train":
        run_experiment(module_name, exp_name)
    elif task == "tune":
        hyperparametersearch(module_name, exp_name, hyperparameters, {"num_epochs": 1})
