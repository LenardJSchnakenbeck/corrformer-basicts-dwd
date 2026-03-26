# corrformer-basicts-dwd

Integration of the [Corrformer](https://github.com/thuml/Corrformer) by Wu et al. (2023) spatiotemporal forecasting model into the [BasicTS](https://github.com/zezhishao/BasicTS) framework, using real-world weather data from the German Weather Service (DWD) via the [wetterdienst](https://github.com/earthobservations/wetterdienst) API.

---

## Architecture

### Data Pipeline
`dwd_download`
- Fetches measurements, timestamps, and station coordinates (lat/lon/alt) from the wetterdienst API
- Handles missing values
- Produces train/validation/test splits
- Converts data into the BasicTS-compatible format

`dwd_dataset` provides a custom dataset class inheriting from `BasicTSDataset`, integrating the data into the BasicTS training pipeline.

### Model: Corrformer

Corrformer is a deep learning architecture based on the transformer architecture, but features spatial cross-correlation and temporal auto-correlation instead of attention mechanisms.
The original implementation was adapted to work with BasicTS and some modules were replaced by their existing BasicTS counterparts (e.g. `MovingAverageDecomposition`).

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.
```
uv sync
```
Installs all dependencies defined in `pyproject.toml`, including PyTorch with CUDA 12.1 support.

> For CPU-only or a different CUDA version, update the index URL in `pyproject.toml` under `[[tool.uv.index]]`.

---

## Usage
```
python prediction_Corrformer.py
```

---

## Credits

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). *Corrformer: Interpretable multivariate time series forecasting with cross-correlation*. Nature Machine Intelligence.

---

## Topics
`time-series` `weather-forecasting` `transformers` `pytorch` `dwd` `machine-learning` `spatiotemporal` `corrformer` `basicts`
