"""
Simple profiling decorator for timing function calls in JAX

Works with both regular Python functions and JAX JIT-compiled functions.
For JIT functions, times include compilation on first call.
"""
import time
import functools
from collections import defaultdict
from typing import Dict, Callable
import threading

import jax
import numpy as np


class ProfileStats:
    """Thread-safe profiling statistics collector."""

    def __init__(self):
        self.stats = defaultdict(list)
        self.lock = threading.Lock()

    def add_timing(self, func_name: str, duration: float):
        """Add a timing measurement for a function."""
        with self.lock:
            self.stats[func_name].append(duration)

    def get_summary(self) -> Dict:
        """Get summary statistics for all profiled functions."""
        with self.lock:
            summary = {}
            for func_name, times in self.stats.items():
                n = len(times)
                if n > 0:
                    total = sum(times)
                    summary[func_name] = {
                        'count': n,
                        'total_time': total,
                        'avg_time': total / n,
                        'median_time': np.median(times),
                        'min_time': min(times),
                        'max_time': max(times)
                    }
                else:
                    summary[func_name] = {
                        'count': 0,
                        'total_time': 0,
                        'avg_time': 0,
                        'median_time': 0,
                        'min_time': 0,
                        'max_time': 0
                    }
            return summary

    def print_summary(self):
        """Print formatted profiling summary."""
        summary = self.get_summary()
        print("\n" + "=" * 100)
        print("PROFILING SUMMARY")
        print("=" * 100)
        print(f"{'Function':<40} {'Count':<8} {'Total(s)':<10} {'Avg(ms)':<10} {'Median(ms)':<12} {'Min(ms)':<10} {'Max(ms)':<10}")
        print("-" * 100)

        # Sort by total time descending
        sorted_funcs = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)

        for func_name, stats in sorted_funcs:
            print(f"{func_name:<40} {stats['count']:<8} {stats['total_time']:<10.3f} "
                  f"{stats['avg_time']*1000:<10.2f} {stats['median_time']*1000:<12.2f} "
                  f"{stats['min_time']*1000:<10.2f} {stats['max_time']*1000:<10.2f}")

        print("=" * 100)

    def clear(self):
        """Clear all profiling statistics."""
        with self.lock:
            self.stats.clear()


# Global profiler instance
profiler = ProfileStats()


def profile(func_name: str = None) -> Callable:
    """
    Decorator to profile function execution time.

    Works with both regular Python functions and JAX JIT-compiled functions.
    For JIT functions, the timing includes compilation on the first call.

    Usage:
        @profile()
        def my_function():
            pass

        @profile("custom_name")
        def another_function():
            pass

        # Can also be used with JAX JIT
        @jax.jit
        @profile("jitted_function")
        def jitted_func(x):
            return x ** 2

    Args:
        func_name: Optional custom name for the function in profiling output.
                   If None, uses module.function_name.

    Returns:
        Decorated function that records execution time.
    """
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                profiler.add_timing(name, duration)

        return wrapper
    return decorator


def profile_jax(func_name: str = None) -> Callable:
    """
    Decorator to profile function calls inside JAX traced code (vmap, scan, jit).

    Uses jax.debug.callback to capture timing from inside compiled JAX code.
    Callbacks are executed in order for sequential operations (scan).

    Usage:
        @profile_jax("my_function")
        def my_function(x):
            return x ** 2

        # Works inside vmap, scan, etc.
        result = jax.vmap(my_function)(batch_x)

        # Can also be used with jax.lax.scan
        result = jax.lax.scan(my_function, init, xs)

    Args:
        func_name: Optional custom name for the function in profiling output.
                   If None, uses module.function_name.

    Returns:
        Decorated function that profiles calls inside JAX traced code.
    """
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"
        start_times = []  # Stack for handling nested/sequential calls
        lock = threading.Lock()  # Thread safety for the stack

        def _push_start():
            """Record start time."""
            with lock:
                start_times.append(time.time())

        def _pop_and_record():
            """Record duration and add to profiler."""
            with lock:
                if start_times:
                    start = start_times.pop()
                    duration = time.time() - start
                    profiler.add_timing(name, duration)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            jax.debug.callback(_push_start)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                jax.debug.callback(_pop_and_record)

        return wrapper
    return decorator


# Convenience functions
def print_profile_summary():
    """Print the profiling summary."""
    profiler.print_summary()


def clear_profile_stats():
    """Clear all profiling statistics."""
    profiler.clear()


def get_profile_stats() -> Dict:
    """
    Get raw profiling statistics.

    Returns:
        Dictionary mapping function names to their statistics.
    """
    return profiler.get_summary()