from typing import Optional

from flax import nnx
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def sharded_init(init: nnx.Initializer, spec: P, mesh: Optional[Mesh]) -> nnx.Initializer:
    """Create a sharded initializer if mesh is provided, otherwise return the original initializer.

    Args:
        init (nnx.Initializer): The initializer to shard.
        spec (P): The sharding specification.
        mesh (Optional[Mesh]): The mesh to shard the initializer on.

    Returns:
        nnx.Initializer: The possibly sharded initializer.
    """
    return nnx.with_partitioning(init, spec) if mesh is not None else init
