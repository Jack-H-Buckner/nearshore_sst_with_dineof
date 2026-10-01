"""The `nearshore-sst` command-line interface."""

from __future__ import annotations

import json

import pytest
import yaml

import synthetic_cube as SC
from nearshore_sst import cli


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    src = root / "source.zarr"
    SC.build(src)
    user = SC.region_user_config(src, root / "out", loop={"max_iter": 2})
    cfg = root / "region.yaml"
    cfg.write_text(yaml.safe_dump(json.loads(json.dumps(user, default=str))))
    return root, src, cfg


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for c in ("init", "check", "offsets", "dineof", "validate", "all", "status", "figures",
              "fetch-insitu"):
        assert c in out


def test_init_writes_a_loadable_config(tmp_path):
    out = tmp_path / "region.new.yaml"
    assert cli.main(["init", "new_area", "--source", "data/x.zarr", "--insitu", "x.nc",
                     "--out", str(out)]) == 0
    cli._bootstrap()
    import region as R
    rc = R.load_region_config(out, resolve=False)
    assert rc["region"]["aoi"] == "new_area"
    assert str(rc["region"]["source"]) == "data/x.zarr"
    assert str(rc["validation"]["insitu"]) == "x.nc"
    with pytest.raises(SystemExit, match="exists"):
        cli.main(["init", "new_area", "--source", "y", "--out", str(out)])


def test_check(setup, capsys, tmp_path):
    root, src, cfg = setup
    assert cli.main(["check", "--config", str(cfg)]) == 0
    assert "OK" in capsys.readouterr().out
    bad = yaml.safe_load(cfg.read_text())
    bad["sensors"]["eco"]["sst"] = "no_such_channel"
    bcfg = tmp_path / "bad.yaml"
    bcfg.write_text(yaml.safe_dump(bad))
    assert cli.main(["check", "--config", str(bcfg)]) == 1
    assert "no_such_channel" in capsys.readouterr().out


def test_stages_and_status(setup, capsys):
    root, src, cfg = setup
    assert cli.main(["status", "-c", str(cfg)]) == 0
    assert "stage 1  not run" in capsys.readouterr().out
    assert cli.main(["offsets", "-c", str(cfg), "--no-figures"]) == 0
    assert cli.main(["dineof", "-c", str(cfg), "--no-figures", "--max-iter", "2"]) == 0
    assert cli.main(["status", "-c", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "stage 1  done" in out and "stage 2  done" in out and "stage 3  not run" in out
    assert (root / "out" / "synthetic" / "stage2_dineof" / "synthetic_dineof.zarr").exists()
    assert cli.main(["figures", "-c", str(cfg)]) == 0
    assert (root / "out" / "synthetic" / "stage2_dineof" / "figures" / "eofs.png").exists()
