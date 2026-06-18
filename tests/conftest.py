import sys
from pathlib import Path

import jax
import pytest
from absl import flags
from dotenv import load_dotenv
from jax.experimental import mesh_utils
from jax.sharding import Mesh

_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

_has_accelerator = any(d.platform in ("gpu", "tpu") for d in jax.devices())


def pytest_configure(config):
    config.addinivalue_line("markers", "tokamax: requires GPU or TPU accelerator")
    config.addinivalue_line("markers", "slow: downloads large model checkpoints from HuggingFace")


def pytest_collection_modifyitems(items):
    skip = pytest.mark.skip(reason="no GPU/TPU available")
    for item in items:
        if item.get_closest_marker("tokamax") and not _has_accelerator:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def jimm_testing_setup():
    """Parse absl flags to prevent absl from choking on pytest flags."""
    flags.FLAGS(sys.argv[:1])
    devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
    mesh = Mesh(devices, ("data", "fsdp"))
    jax.set_mesh(mesh)
    yield
