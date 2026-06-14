import sys

import jax
import pytest
from absl import flags
from jax.experimental import mesh_utils
from jax.sharding import Mesh


@pytest.fixture(scope="session", autouse=True)
def jimm_testing_setup():
    """Parse absl flags to prevent absl from choking on pytest flags."""
    flags.FLAGS(sys.argv[:1])
    devices = mesh_utils.create_device_mesh((jax.device_count(), 1))
    mesh = Mesh(devices, ("data", "fsdp"))
    jax.set_mesh(mesh)
    yield
