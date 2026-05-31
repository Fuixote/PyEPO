import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from run import utils
from run.fixed_sp_data import (
    dataset_path,
    load_split_dataset,
    load_split_for_config,
    save_split_dataset,
)


class FixedShortestPathDataTest(unittest.TestCase):
    def make_config(self, data_path):
        return SimpleNamespace(
            data_path=data_path,
            prob="sp",
            grid=(5, 5),
            data=2,
            feat=3,
            deg=1,
            noise=0.5,
            seed=7,
            test_size=1000,
        )

    def test_dataset_path_encodes_shortest_path_configuration(self):
        cfg = self.make_config("../data")

        path = dataset_path(cfg)

        self.assertEqual(
            str(path),
            "../data/sp/h5w5/train2-test1000-p3-d1-e0.5-seed7.npz",
        )

    def test_save_and_load_split_dataset_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.make_config(tmp)
            path = dataset_path(cfg)
            x_train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            c_train = np.array([[1.5, 2.5], [3.5, 4.5]])
            x_test = np.array([[7.0, 8.0, 9.0]])
            c_test = np.array([[5.5, 6.5]])
            metadata = {
                "problem": "sp",
                "grid": [5, 5],
                "train_size": 2,
                "test_size": 1000,
                "feat": 3,
                "deg": 1,
                "noise": 0.5,
                "seed": 7,
            }

            save_split_dataset(path, x_train, c_train, x_test, c_test, metadata)
            loaded = load_split_dataset(path)

            np.testing.assert_array_equal(loaded.x_train, x_train)
            np.testing.assert_array_equal(loaded.c_train, c_train)
            np.testing.assert_array_equal(loaded.x_test, x_test)
            np.testing.assert_array_equal(loaded.c_test, c_test)
            self.assertEqual(loaded.metadata, metadata)

    def test_missing_fixed_dataset_error_names_generator(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.make_config(tmp)

            with self.assertRaisesRegex(FileNotFoundError, "generate_sp_datasets.py"):
                load_split_for_config(cfg)

    def test_gen_data_loads_fixed_split_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.make_config(tmp)
            cfg.fixed_data = True
            path = dataset_path(cfg)
            x_train = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            c_train = np.array([[1.5, 2.5], [3.5, 4.5]])
            x_test = np.array([[7.0, 8.0, 9.0]])
            c_test = np.array([[5.5, 6.5]])
            save_split_dataset(path, x_train, c_train, x_test, c_test, {})

            loaded = utils.genData(cfg)

            np.testing.assert_array_equal(loaded.x_train, x_train)
            np.testing.assert_array_equal(loaded.c_train, c_train)
            np.testing.assert_array_equal(loaded.x_test, x_test)
            np.testing.assert_array_equal(loaded.c_test, c_test)


if __name__ == "__main__":
    unittest.main()
