import sys
from pathlib import Path

import jax
import pytest
from absl import flags
from dotenv import load_dotenv
from jax.experimental import mesh_utils
from jax.sharding import Mesh

load_dotenv(Path(__file__).parent.parent / ".env")

_has_accelerator = any(d.platform in ("gpu", "tpu") for d in jax.devices())


def pytest_configure(config):
    config.addinivalue_line("markers", "tokamax: requires GPU or TPU accelerator")


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="no GPU/TPU available")
    for item in items:
        if any(kw in item.name for kw in ("tokamax", "long_context", "autotune")):
            if not _has_accelerator:
                item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def jimm_testing_setup():
    """Parse absl flags to prevent absl from choking on pytest flags."""
    flags.FLAGS(sys.argv[:1])
    devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
    mesh = Mesh(devices, ("data", "fsdp"))
    jax.set_mesh(mesh)
    yield
