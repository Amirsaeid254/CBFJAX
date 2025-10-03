"""
JAX utility functions for CBF computations.
Optimized for performance and JIT compilation.
"""

import jax
import jax.numpy as jnp
import functools
from typing import Callable, List, Dict




def softmax(x, rho, conservative=True, dim=0):
    res = (1.0 / rho) * jax.nn.logsumexp(rho * x, axis=dim)
    if conservative:
        res = res - jnp.log(x.shape[dim]) / rho
    return res

def softmin(x, rho, conservative=False, dim=0):
    return softmax(x=x, rho=-rho, conservative=conservative, dim=dim)




def lie_deriv(func: Callable, field: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Lie derivative of func along vector field.

    JAX Optimization: Direct jax.grad - no manual graph management needed

    Args:
        func: Scalar function R^n -> R
        field: Vector field R^n -> R^n or R^n -> R^{n×m}
        x: State vector (n,)

    Returns:
        Lie derivative L_field(func)(x)
    """
    # Get gradient of func at x
    grad_func = jax.grad(func)
    func_deriv = grad_func(x)  # Shape: (n,)

    # Get field value at x
    field_val = field(x)  # Shape: (n,) or (n,m)

    return lie_deriv_from_values(func_deriv, field_val)


def lie_deriv_from_values(func_deriv: jnp.ndarray, field_val: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Lie derivative from pre-computed gradient and field values.

    JAX Optimization: Direct einsum operations, more efficient than loops

    Args:
        func_deriv: Gradient of function (n,)
        field_val: Field value (n,) or (n,m)

    Returns:
        Lie derivative value (scalar or (m,))
    """
    if field_val.ndim == 1:
        # Vector field case: (n,) @ (n,) -> scalar
        return jnp.dot(func_deriv, field_val)
    elif field_val.ndim == 2:
        # Control matrix case: (n,) @ (n,m) -> (m,)
        return func_deriv @ field_val
    else:
        raise ValueError(f"Field dimension {field_val.ndim} not supported")


def get_func_deriv(func: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """
    Get gradient of function at point x.

    Args:
        func: Function R^n -> R or R^n -> R^m
        x: Input point (n,)

    Returns:
        Gradient(s) - (n,) for scalar func, (m,n) for vector func
    """
    func_val = func(x)

    if func_val.ndim == 0:  # Scalar function
        return jax.grad(func)(x)
    else:  # Vector function - compute jacobian
        return jax.jacrev(func)(x)


def make_higher_order_lie_deriv_series(func: Callable, field: Callable, deg: int) -> List[Callable]:
    """
    Generate series of higher-order Lie derivatives.

    Args:
        func: Initial function
        field: Vector field
        deg: Degree of derivatives to compute

    Returns:
        List of functions [func, L_f(func), L_f^2(func), ...]
    """
    derivatives = [func]

    for i in range(deg):
        # Create next derivative using pure functional composition
        prev_deriv = derivatives[i]
        next_deriv = functools.partial(lie_deriv, prev_deriv, field)
        derivatives.append(next_deriv)

    return derivatives


def match_dim(res: jnp.ndarray, x: jnp.ndarray, ) -> jnp.ndarray:
    """
    Match dimensions exactly.

    Args:
        x: Original input array
        res: Result array to match dimensions

    Returns:
        Result with dimension matching applied
    """
    if x.ndim == 1 and res.ndim == 1:
        return res
    # Check if dimensions need to be matched
    if x.ndim == 1 and res.ndim == 2:
        return jnp.squeeze(res, axis=0)  # Remove first dimension
    if x.ndim == 2 and res.ndim == 1:
        return res.reshape(-1, 1)  # Add dimension at end

    return res

def apply_and_match_dim(func: Callable, x: jnp.ndarray) -> jnp.ndarray:
    """
    Apply function and match dimensions.

    Args:
        func: Function to apply
        x: Input array

    Returns:
        Result with dimension matching applied
    """
    # Apply the function to the input tensor
    res = func(x)
    return match_dim(res, x)

def ensure_batch_dim(x: jnp.ndarray, target_ndim: int = 2) -> jnp.ndarray:
    """
    Ensure array has at least target number of dimensions.

    Args:
        x: Input array
        target_ndim: Target number of dimensions (default 2)

    Returns:
        Array with at least target_ndim dimensions
    """
    while x.ndim < target_ndim:
        x = jnp.expand_dims(x, 0)
    return x



def apply_and_batchize(func: Callable, x: jnp.ndarray):
    """ALWAYS use vmap for consistent behavior"""
    x_batched = jnp.atleast_2d(x)  # Ensure batch dim
    batched_func = jax.vmap(func)    # Always vmap
    return apply_and_match_dim(batched_func, x_batched)


def apply_and_batchize_tuple(func: Callable, x: jnp.ndarray):
    x_batched = ensure_batch_dim(x)  # Ensure batch dim
    batched_func = jax.vmap(func)  # Always vmap
    result = batched_func(x_batched)  # Apply function - should return tuple

    # Apply dimension matching to each element of the tuple

    return tuple(match_dim(item, x_batched) for item in result)

def higher_order_lie_deriv(func: Callable, field: Callable, order: int) -> Callable:
    """
    Compute higher-order Lie derivative directly.

    JAX Optimization: Automatic differentiation for arbitrary orders
    """
    result_func = func
    for _ in range(order):
        result_func = functools.partial(lie_deriv, result_func, field)
    return result_func



def update_dict_no_overwrite(target_dict: Dict, update_dict: Dict) -> Dict:
    """Helper function to update dict without overwriting existing keys."""
    result = target_dict.copy()
    for key, value in update_dict.items():
        if key not in result:
            result[key] = value
    return result


# ============================================================================
# Geometric Barrier Functions
# ============================================================================

def vectorize_tensors(arr):
    if isinstance(arr, jnp.ndarray):
        return jnp.expand_dims(arr, 0) if arr.ndim == 1 else arr
    elif isinstance(arr, (list, tuple)):
        arr = jnp.array(arr)
        return jnp.expand_dims(arr, 0) if arr.ndim == 1 else arr
    else:
        return jnp.array(arr)


def rotate_tensors(points: jnp.ndarray, center: jnp.ndarray, angle_rad: float) -> jnp.ndarray:
    """
    Rotate points around center by angle_rad.

    Args:
        points: Points to rotate (..., 2) or (..., n) where first 2 dims are spatial
        center: Center of rotation (..., 2) or (..., n)
        angle_rad: Rotation angle in radians

    Returns:
        Rotated points with same shape as input
    """
    center_size = center.shape[-1] if center.ndim > 0 else 2
    rotation_matrix = jnp.array([
        [jnp.cos(angle_rad), -jnp.sin(angle_rad)],
        [jnp.sin(angle_rad), jnp.cos(angle_rad)]
    ])

    # Rotate only the first 2 dimensions (spatial coordinates)
    points_2d = points[:2]
    center_2d = center[:2]

    rotated_xy = jnp.dot(points_2d - center_2d, rotation_matrix.T) + center_2d

    # Concatenate with remaining dimensions if they exist
    if points.shape[-1] > 2:
        return jnp.concatenate([rotated_xy, points[2:center_size]])
    else:
        return rotated_xy


def make_circle_barrier_functional(center, radius):
    """
    Create circle barrier function.

    Args:
        center: Circle center (2,)
        radius: Circle radius (scalar)

    Returns:
        Barrier function that computes distance to circle boundary
    """
    center = jnp.array(center)

    def circle_barrier(x):
        pos = x[:2]  # Extract position components
        return jnp.linalg.norm(pos - center) / radius - 1.0

    return circle_barrier


def make_ellipse_barrier_functional(center, A):
    """
    Create ellipse barrier function.

    Args:
        center: Ellipse center (2,)
        A: Ellipse matrix (2, 2)

    Returns:
        Barrier function for ellipse
    """
    center = jnp.array(center)
    A = jnp.array(A)

    def ellipse_barrier(x):
        pos = x[:2]  # Extract position components
        diff = pos - center

        # Compute quadratic form: (x-c)^T A (x-c)
        quadratic_form = diff.T @ A @ diff

        return 1 - quadratic_form

    return ellipse_barrier


def make_norm_rectangular_barrier_functional(center, size, rotation=0.0, p=20):
    """
    Create norm-based rectangular barrier function.

    Args:
        center: Rectangle center (2,)
        size: Rectangle half-sizes (2,)
        rotation: Rotation angle in radians
        p: Norm parameter (higher = more rectangular)

    Returns:
        Barrier function for norm-based rectangle
    """
    center = jnp.array(center)
    size = jnp.array(size)

    def norm_rect_barrier(x):
        # Rotate points to rectangle's local frame
        rotated_points = rotate_tensors(x, center, -rotation)

        # Compute normalized distance from center
        normalized_dist = (rotated_points[:2] - center) / size

        # Compute p-norm distance
        return jnp.linalg.norm(normalized_dist, ord=p) - 1.0

    return norm_rect_barrier


def make_affine_rectangular_barrier_functional(center, size, rotation=0.0, smooth=False, rho=40):
    """
    Create affine rectangular barrier function.

    Args:
        center: Rectangle center (2,)
        size: Rectangle half-sizes (2,)
        rotation: Rotation angle in radians
        smooth: Whether to use smooth maximum
        rho: Smoothness parameter for smooth maximum

    Returns:
        Barrier function for affine rectangle
    """
    center = jnp.array(center)
    size = jnp.array(size)

    # Define normals for axis-aligned rectangle
    A = jnp.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    b = jnp.array([center[0] + size[0], -center[0] + size[0],
                   center[1] + size[1], -center[1] + size[1]])

    def affine_rect_barrier(x):

        # Rotate points to rectangle's local frame
        rotated_points = rotate_tensors(x[:2], center, -rotation)

        # Compute affine constraints: A * x - b
        constraints = jnp.dot(rotated_points, A.T) - b

        if smooth:
            return softmax(constraints, rho=rho, dim=-1)
        else:
            return jnp.max(constraints)

    return affine_rect_barrier


def make_norm_rectangular_boundary_functional(center, size, rotation=0.0, p=20):
    """
    Create norm-based rectangular boundary function (safe inside).

    Args:
        center: Rectangle center (2,)
        size: Rectangle half-sizes (2,)
        rotation: Rotation angle in radians
        p: Norm parameter

    Returns:
        Boundary function (negative of barrier)
    """
    barrier_func = make_norm_rectangular_barrier_functional(center, size, rotation, p)

    def boundary_func(x):
        return -barrier_func(x)

    return boundary_func


def make_affine_rectangular_boundary_functional(center, size, rotation=0.0, smooth=False, rho=40):
    """
    Create affine rectangular boundary function (safe inside).

    Args:
        center: Rectangle center (2,)
        size: Rectangle half-sizes (2,)
        rotation: Rotation angle in radians
        smooth: Whether to use smooth maximum
        rho: Smoothness parameter

    Returns:
        Boundary function (negative of barrier)
    """
    barrier_func = make_affine_rectangular_barrier_functional(center, size, rotation, smooth, rho)

    def boundary_func(x):
        return -barrier_func(x)

    return boundary_func


def make_box_barrier_functionals(bounds, idx):
    # Convert bounds to scalars to avoid capturing unhashable objects
    lb, ub = float(bounds[0]), float(bounds[1])
    idx_val = int(idx)

    def create_lower_barrier(lower_bound, index):
        def lower_barrier(x):
            return x[index] - lower_bound
        return lower_barrier

    def create_upper_barrier(upper_bound, index):
        def upper_barrier(x):
            return upper_bound - x[index]
        return upper_barrier

    return [
        create_lower_barrier(lb, idx_val),
        create_upper_barrier(ub, idx_val),
    ]


def make_linear_alpha_function_form_list_of_coef(coef_list):
    # Convert to tuple to avoid unhashable list in closure
    coef_tuple = tuple(coef_list)

    def create_linear_alpha(c):
        def alpha_func(x):
            return float(c) * x  # Ensure scalar

        return alpha_func

    return [create_linear_alpha(c) for c in coef_tuple]


def make_cubic_alpha_function_form_list_of_coef(coef_list):
    """
    Create cubic alpha functions from coefficient list.

    Args:
        coef_list: List of coefficients for alpha functions

    Returns:
        List of alpha functions [c1*x^3, c2*x^3, ...]
    """
    def create_cubic_alpha(c):
        def alpha_func(x):
            return c * x**3
        return alpha_func

    return [create_cubic_alpha(c) for c in coef_list]


def make_tanh_alpha_function_form_list_of_coef(coef_list):
    """
    Create tanh alpha functions from coefficient list.

    Args:
        coef_list: List of coefficients for alpha functions

    Returns:
        List of alpha functions [c1*tanh(x), c2*tanh(x), ...]
    """
    def create_tanh_alpha(c):
        def alpha_func(x):
            return c * jnp.tanh(x)
        return alpha_func

    return [create_tanh_alpha(c) for c in coef_list]


