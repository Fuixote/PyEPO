#!/usr/bin/env python
# coding: utf-8
"""
Fixed synthetic datasets for shortest-path experiments.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import pyepo


DEFAULT_TEST_SIZE = 1000


@dataclass(frozen=True)
class SplitDataset:
    x_train: np.ndarray
    c_train: np.ndarray
    x_test: np.ndarray
    c_test: np.ndarray
    metadata: dict


def _test_size(config):
    return getattr(config, "test_size", DEFAULT_TEST_SIZE)


def dataset_path(config):
    grid = tuple(config.grid)
    path = Path(config.data_path)
    path = path / config.prob / "h{}w{}".format(*grid)
    filename = (
        "train{}-test{}-p{}-d{}-e{}-seed{}.npz".format(
            config.data,
            _test_size(config),
            config.feat,
            config.deg,
            config.noise,
            config.seed,
        )
    )
    return path / filename


def metadata_for_config(config):
    return {
        "problem": config.prob,
        "grid": list(config.grid),
        "train_size": config.data,
        "test_size": _test_size(config),
        "feat": config.feat,
        "deg": config.deg,
        "noise": config.noise,
        "seed": config.seed,
    }


def generate_split_for_config(config):
    test_size = _test_size(config)
    x, c = pyepo.data.shortestpath.genData(
        config.data + test_size,
        config.feat,
        tuple(config.grid),
        deg=config.deg,
        noise_width=config.noise,
        seed=config.seed,
    )
    x_train, x_test, c_train, c_test = train_test_split(
        x,
        c,
        test_size=test_size,
        random_state=config.seed,
    )
    return SplitDataset(
        x_train=x_train,
        c_train=c_train,
        x_test=x_test,
        c_test=c_test,
        metadata=metadata_for_config(config),
    )


def save_split_dataset(path, x_train, c_train, x_test, c_test, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=x_train,
        c_train=c_train,
        x_test=x_test,
        c_test=c_test,
        metadata=json.dumps(metadata, sort_keys=True),
    )


def save_split_for_config(config, split):
    save_split_dataset(
        dataset_path(config),
        split.x_train,
        split.c_train,
        split.x_test,
        split.c_test,
        split.metadata,
    )


def load_split_dataset(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = data["metadata"].item()
        return SplitDataset(
            x_train=data["x_train"],
            c_train=data["c_train"],
            x_test=data["x_test"],
            c_test=data["c_test"],
            metadata=json.loads(metadata_raw),
        )


def load_split_for_config(config):
    path = dataset_path(config)
    if not path.exists():
        raise FileNotFoundError(
            "Fixed dataset not found: {}. Run `python3 generate_sp_datasets.py` "
            "from the PyEPO checkout first.".format(path)
        )
    return load_split_dataset(path)
