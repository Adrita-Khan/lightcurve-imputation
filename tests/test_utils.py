"""Tests for utility modules."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config, merge_configs
from src.utils.io import save_results, load_results, save_table
from src.utils.seeds import make_seed_sequence
from src.utils.logging_utils import get_logger


class TestLoadConfig:
    def test_load_experiment_config(self):
        cfg = load_config("configs/experiment.yml")
        assert "signal" in cfg
        assert "missingness" in cfg
        assert "methods" in cfg

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("does_not_exist.yml")

    def test_fast_test_config(self):
        cfg = load_config("configs/fast_test.yml")
        assert cfg["missingness"]["n_seeds"] < 10


class TestMergeConfigs:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = merge_configs(base, override)
        assert result["a"] == 1
        assert result["b"] == 99

    def test_nested_merge(self):
        base = {"signal": {"N": 100, "dt": 0.5}}
        override = {"signal": {"N": 200}}
        result = merge_configs(base, override)
        assert result["signal"]["N"] == 200
        assert result["signal"]["dt"] == 0.5

    def test_does_not_mutate_base(self):
        base = {"x": 1}
        override = {"x": 2}
        merge_configs(base, override)
        assert base["x"] == 1


class TestIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        df = pd.DataFrame({"method": ["A", "B"], "rmse": [0.1, 0.2]})
        path = tmp_path / "results.csv"
        save_results(df, path)
        loaded = load_results(path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_creates_parent_dirs(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        path = tmp_path / "subdir" / "results.csv"
        save_results(df, path)
        assert path.exists()

    def test_save_latex_table(self, tmp_path):
        df = pd.DataFrame({"method": ["A"], "rmse": [0.123]})
        path = tmp_path / "table.tex"
        save_table(df, path, fmt="latex", caption="Test", label="tab:test")
        content = path.read_text()
        assert "\\begin{table}" in content
        assert "Test" in content

    def test_save_csv_table(self, tmp_path):
        df = pd.DataFrame({"method": ["A", "B"], "mae": [0.05, 0.10]})
        path = tmp_path / "table.csv"
        save_table(df, path, fmt="csv")
        loaded = pd.read_csv(path)
        assert list(loaded.columns) == ["method", "mae"]

    def test_unknown_format_raises(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError):
            save_table(df, tmp_path / "out.xyz", fmt="xyz")


class TestSeeds:
    def test_length(self):
        seeds = make_seed_sequence(30, base_seed=42)
        assert len(seeds) == 30

    def test_all_integers(self):
        seeds = make_seed_sequence(10, base_seed=0)
        assert all(isinstance(s, int) for s in seeds)

    def test_reproducible(self):
        s1 = make_seed_sequence(20, base_seed=7)
        s2 = make_seed_sequence(20, base_seed=7)
        assert s1 == s2

    def test_different_base_different_seeds(self):
        s1 = make_seed_sequence(5, base_seed=1)
        s2 = make_seed_sequence(5, base_seed=2)
        assert s1 != s2

    def test_unique_seeds(self):
        seeds = make_seed_sequence(30, base_seed=42)
        assert len(set(seeds)) == len(seeds)


class TestLogger:
    def test_returns_logger(self):
        import logging
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self):
        logger = get_logger("my_module")
        assert logger.name == "my_module"
