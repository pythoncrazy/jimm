import sys
from absl import flags
import pytest


@pytest.fixture(scope='session', autouse=True)
def jimm_testing_setup():
    """Parse absl flags to prevent absl from choking on pytest flags."""
    flags.FLAGS(sys.argv[:1])
    yield
