"""
Upgrading of outdated configuration files ("beat update").
"""

import os

import pytest

from beat import config as bconfig
from beat.upgrade import upgrade_config_file


@pytest.fixture
def old_config_file(geodetic_project, tmp_path):
    """
    Config of the synthetic geodetic project with attributes of older beat
    versions.
    """
    project = geodetic_project.copy(tmp_path)
    fn = os.path.join(project.project_dir, "config_geometry.yaml")
    with open(fn) as f:
        text = f.read()

    old = "geodetic_config: !beat.GeodeticConfig\n"
    assert old in text
    text = text.replace(old, old + "  calc_data_cov: true\n  fit_plane: false\n", 1)

    old = "    - !beat.EulerPoleConfig\n"
    assert old in text
    text = text.replace(old, old + "      blacklist: [G01, G02]\n", 1)

    with open(fn, "w") as f:
        f.write(text)

    return project.project_dir, fn


def test_old_config_needs_update(old_config_file):
    project_dir, _ = old_config_file
    with pytest.raises(bconfig.ConfigNeedsUpdatingError):
        bconfig.load_config(project_dir, "geometry")


def test_upgrade_structure(old_config_file):
    project_dir, fn = old_config_file
    upgrade_config_file(fn, diff=False, update=["structure"])

    config = bconfig.load_config(project_dir, "geometry")
    config.validate()
    euler_pole = config.geodetic_config.corrections_config.euler_poles[0]
    assert euler_pole.station_blacklist == ["G01", "G02"]

    with open(fn) as f:
        text = f.read()
    assert "calc_data_cov" not in text
    assert "fit_plane" not in text


def test_upgrade_diff_does_not_write(old_config_file, capsys):
    project_dir, fn = old_config_file
    with open(fn) as f:
        before = f.read()

    upgrade_config_file(fn, diff=True, update=["structure"])

    diff = capsys.readouterr().out
    assert "-  calc_data_cov: true" in diff
    assert "-      blacklist:" in diff
    assert "+      station_blacklist:" in diff
    with open(fn) as f:
        assert f.read() == before


def test_upgrade_hypers(geodetic_project, tmp_path):
    project = geodetic_project.copy(tmp_path)
    config = project.config
    config.problem_config.hyperparameters.pop("h_GNSS")
    bconfig.dump_config(config)

    fn = os.path.join(project.project_dir, "config_geometry.yaml")
    upgrade_config_file(fn, diff=False, update=["hypers"])
    config = bconfig.load_config(project.project_dir, "geometry")
    assert set(config.problem_config.hyperparameters) == {"h_SAR", "h_GNSS"}


def test_upgrade_invalid_update(old_config_file):
    _, fn = old_config_file
    with pytest.raises(TypeError):
        upgrade_config_file(fn, diff=False, update=["not_existing"])
