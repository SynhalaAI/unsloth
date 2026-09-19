# SPDX-License-Identifier: AGPL-3.0-only
"""Contract tests for mergekit installation in studio/setup.sh."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_setup_sh_contains_mergekit_install_logic():
    script = (ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    assert "_setup_install_mergekit()" in script, "setup.sh must define _setup_install_mergekit"
    assert "UNSLOTH_DISABLE_MERGEKIT" in script, "must allow disabling mergekit installation"
    assert "importlib.util.find_spec('mergekit')" in script, "must check if mergekit is already installed"
    assert 'fast_install "mergekit"' in script, "must install mergekit via fast_install"
    assert 'run_quiet_no_exit "install mergekit"' in script, "must not abort setup if mergekit install fails"
    assert "_uv_offline_requested" in script, "must respect offline mode"
    assert "_setup_install_mergekit\nfi" in script or "_setup_install_mergekit\n" in script, "must invoke _setup_install_mergekit"
