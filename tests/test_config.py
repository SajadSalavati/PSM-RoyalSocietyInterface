"""Tests for the experiment configuration system."""

import glob

import pytest

from psm_surrogate.config import ExperimentConfig, load_config

CONFIG_FILES = sorted(glob.glob("configs/*.yaml"))


def test_configs_exist():
    assert len(CONFIG_FILES) == 12


@pytest.mark.parametrize("path", CONFIG_FILES)
def test_load_config(path):
    cfg = load_config(path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model in {"Base", "Sten"}
    assert cfg.bc_name in {"Mid", "Wall"}
    assert cfg.num_modes >= 1


def test_derived_properties():
    cfg = load_config("configs/base_mid_u.yaml")
    assert cfg.input_size == cfg.num_modes + 3
    assert cfg.output_size == cfg.num_modes
    assert cfg.weight_filename == "LSTMNet_Base_Mid_v_x.pt"
    assert cfg.pod_result_filename == "POD_data_Base_Mid_v_x.npz"
    assert cfg.data_path.endswith("Base/Mid/rylScty_patientSpec_BaseModel_miduvw.npz")


def test_unknown_keys_go_to_extra(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "name: x\nmodel: Base\nbc_name: Mid\nvariable_name: v_x\n"
        "field_key: v_x\nnpz_filename: a.npz\nnum_modes: 5\nmystery: 42\n"
    )
    cfg = load_config(str(p))
    assert cfg.extra == {"mystery": 42}
