"""
Configuration settings for CBF-JAX.
"""

import jax
import jax.numpy as jnp

# Default precision for all CBF-JAX computations
# CBF methods require high precision for numerical stability
DEFAULT_DTYPE = jnp.float64


def get_default_dtype():
    """Get the current default dtype for CBF-JAX computations."""
    return DEFAULT_DTYPE


def set_default_dtype(dtype):
    """
    Set the default dtype for CBF-JAX computations.

    Parameters:
        dtype: jnp.dtype
            The desired default dtype (e.g., jnp.float32, jnp.float64)

    Example:
      import cbfjax
      cbfjax.set_default_dtype(jnp.float32)  # Use single precision
      cbfjax.set_default_dtype(jnp.float64)  # Use double precision (default)
    """
    global DEFAULT_DTYPE
    if not hasattr(dtype, 'dtype'):  # Check if it's a valid JAX dtype
        try:
            # Try to convert to JAX array to validate dtype
            test_array = jnp.array(0.0, dtype=dtype)
            dtype = test_array.dtype
        except (TypeError, ValueError):
            raise ValueError(f"dtype must be a valid JAX dtype, got {type(dtype)}")
    DEFAULT_DTYPE = dtype


def configure_jax(platform=None, enable_x64=True, debug_nans=False, disable_jit=False):
    """
    Configure JAX settings for CBF computations.

    Must be called before the first JAX computation to take effect
    (except disable_jit, which can be toggled later).

    Parameters:
        platform: str or None
            Platform to use ("cpu", "gpu", or "tpu"). None keeps JAX's
            automatic platform selection.
        enable_x64: bool
            Whether to enable 64-bit precision (recommended for CBF)
        debug_nans: bool
            Whether to enable NaN debugging (useful for development)
        disable_jit: bool
            If True, every jax.jit / eqx.filter_jit becomes a no-op so
            you can pdb, print, and see Python stack traces.

    Example:
        configure_jax(platform="gpu", enable_x64=True)
        configure_jax(platform="cpu", enable_x64=True, disable_jit=True)
    """
    if platform is not None:
        jax.config.update("jax_platform_name", platform)
    jax.config.update("jax_enable_x64", enable_x64)
    jax.config.update("jax_disable_jit", bool(disable_jit))

    if debug_nans:
        jax.config.update("jax_debug_nans", True)

    # Update default dtype based on x64 setting
    global DEFAULT_DTYPE
    if enable_x64:
        DEFAULT_DTYPE = jnp.float64
    else:
        DEFAULT_DTYPE = jnp.float32


def get_jax_config():
    """
    Get current JAX configuration relevant to CBF computations.

    Returns:
        dict: Current JAX configuration
    """
    return {
        "platform": jax.default_backend(),
        "enable_x64": jax.config.jax_enable_x64,
        "default_dtype": DEFAULT_DTYPE,
        "debug_nans": getattr(jax.config, 'jax_debug_nans', False),
        "disable_jit": bool(getattr(jax.config, 'jax_disable_jit', False)),
    }


# Common dtype aliases for convenience
FLOAT32 = jnp.float32
FLOAT64 = jnp.float64
INT32 = jnp.int32
INT64 = jnp.int64