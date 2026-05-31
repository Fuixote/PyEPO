#!/usr/bin/env python
# coding: utf-8
"""
Generate fixed synthetic shortest-path datasets for PyEPO experiments.
"""

import argparse
import itertools
from types import SimpleNamespace

from run import fixed_sp_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../data")
    parser.add_argument("--spgrid", type=int, nargs=2, default=(5, 5))
    parser.add_argument("--feat", type=int, default=5)
    parser.add_argument("--train-sizes", type=int, nargs="+", default=[100, 1000, 5000])
    parser.add_argument("--degs", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--test-size", type=int, default=fixed_sp_data.DEFAULT_TEST_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_config(args, train_size, deg, noise, seed):
    return SimpleNamespace(
        data_path=args.data_path,
        prob="sp",
        grid=tuple(args.spgrid),
        data=train_size,
        feat=args.feat,
        deg=deg,
        noise=noise,
        seed=seed,
        test_size=args.test_size,
    )


def main():
    args = parse_args()
    total = 0
    generated = 0
    skipped = 0
    for train_size, deg, noise, seed in itertools.product(
        args.train_sizes,
        args.degs,
        args.noises,
        args.seeds,
    ):
        cfg = make_config(args, train_size, deg, noise, seed)
        path = fixed_sp_data.dataset_path(cfg)
        total += 1
        if path.exists() and not args.overwrite:
            skipped += 1
            print("Exists: {}".format(path))
            continue
        split = fixed_sp_data.generate_split_for_config(cfg)
        fixed_sp_data.save_split_for_config(cfg, split)
        generated += 1
        print("Saved: {}".format(path))
    print(
        "Datasets complete: total={}, generated={}, skipped={}".format(
            total,
            generated,
            skipped,
        )
    )


if __name__ == "__main__":
    main()
