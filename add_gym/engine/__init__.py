"""
Engine abstraction layer for physics simulation backends.

Engines are instantiated via Hydra using the _target_ field in config files.
No manual factory needed - Hydra's instantiate() handles creation.

Example usage:
    from hydra.utils import instantiate

    # Config contains _target_: gym.engine.genesis_engine.GenesisEngine
    engine = instantiate(config["engine"])
    engine.init(backend="gpu", precision="32")
    scene = engine.create_scene(...)
"""

from add_gym.engine.base_engine import (
    BaseEngine,
    BaseScene,
    BaseEntity,
    BaseJoint,
    BaseLink,
    BaseCamera
)
from add_gym.engine.genesis_engine import GenesisEngine
from add_gym.engine.mjwarp_engine import MJWarpEngine

__all__ = [
    'BaseEngine',
    'BaseScene',
    'BaseEntity',
    'BaseJoint',
    'BaseLink',
    'BaseCamera',
    'GenesisEngine',
    'MJWarpEngine'
]
