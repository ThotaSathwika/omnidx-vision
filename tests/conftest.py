"""Test setup: isolated storage/DB, a synthetic demo dataset and one quickly trained demo model."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="omnidx_test_"))
DATA = TMP / "demo_xray"

cfg = yaml.safe_load((ROOT / "configs" / "demo.yaml").read_text())
cfg["modules"]["untrained_demo"] = dict(cfg["modules"]["demo_xray"], name="Untrained demo module")
(TMP / "config.yaml").write_text(yaml.safe_dump(cfg))

os.environ.update(OMNIDX_CONFIG=str(TMP / "config.yaml"), OMNIDX_MODELS=str(TMP / "models"),
                  OMNIDX_STORAGE=str(TMP / "storage"), OMNIDX_ADMIN_PASSWORD="admin-pass-123")
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def demo_data():
    if not DATA.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_data.py"), "--out", str(DATA),
                        "--patients", "50"], check=True)
    return DATA


@pytest.fixture(scope="session")
def trained_run(demo_data):
    from omnidx.train import main
    return main(["--module", "demo_xray", "--data", str(demo_data), "--epochs", "12", "--patience", "12",
                 "--batch", "16", "--lr", "1e-3", "--no-pretrained", "--freeze-epochs", "0", "--workers", "0",
                 "--fusion"])
