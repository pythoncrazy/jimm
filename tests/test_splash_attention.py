"""Tests for splash attention config and utilities."""

import importlib.util

from flax import nnx

from jimm import SplashAttentionConfig
from jimm.common.splash_attention import create_splash_attention_fn


def test_splash_attention_config_defaults() -> None:
    config = SplashAttentionConfig()
    assert config.enabled is False
    assert config.mask_type == "full"
    assert config.block_q == 128
    assert config.block_kv == 128


def test_splash_attention_config_custom() -> None:
    config = SplashAttentionConfig(
        enabled=True,
        mask_type="causal",
        block_q=64,
        block_kv=256,
    )
    assert config.enabled is True
    assert config.mask_type == "causal"
    assert config.block_q == 64
    assert config.block_kv == 256


def test_create_fn_returns_default_when_disabled() -> None:
    config = SplashAttentionConfig(enabled=False)
    fn = create_splash_attention_fn(config, num_heads=8, head_dim=64)
    assert fn is nnx.dot_product_attention


def test_create_fn_returns_callable_when_enabled_and_tokamax_available() -> None:
    if importlib.util.find_spec("tokamax") is None:
        return
    config = SplashAttentionConfig(enabled=True)
    fn = create_splash_attention_fn(config, num_heads=8, head_dim=64)
    assert callable(fn)
