import os
import time
import tracemalloc


class PerformanceMonitor:
    """A context manager for benchmarking code.

    This class provides a simple way to measure the execution time and peak
    memory usage of a block of code. It is designed to be used with the
    `with` statement.

    Responsibilities:
      * Start and stop timing and memory tracing.
      * Store the runtime in seconds.
      * Store the peak memory usage in megabytes.

    Example:

        .. code-block:: python

            import rayroom as rt

            with rt.analytics.performance.PerformanceMonitor() as monitor:
                # Code to be benchmarked
                result = [i**2 for i in range(1000000)]

            print(f"Execution time: {monitor.runtime_s:.4f} seconds")
            print(f"Peak memory usage: {monitor.peak_memory_mb:.2f} MB")

    """
    def __init__(self):
        self.runtime_s = 0
        self.peak_memory_mb = 0

    def __enter__(self):
        tracemalloc.start()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.runtime_s = self.end_time - self.start_time
        self.peak_memory_mb = peak / 1024**2  # Convert bytes to MB


def plot_performance_results(results, param_name, output_dir, param_is_log=False):
    """Plots and saves performance benchmark results.

    This function generates and saves plots for runtime and peak memory usage
    based on benchmark data. It is useful for visualizing how performance
    varies with a given parameter.

    Responsibilities:
      * Generate a plot for runtime vs. the varied parameter.
      * Generate a plot for memory usage vs. the varied parameter.
      * Save the plots to the specified directory.
      * Handle logarithmic scales for the parameter axis.

    :param results: A dictionary containing the benchmark data. The keys are
                    engine names, and the values are dictionaries where keys
                    are parameter values and values are performance metrics.
    :type results: dict
    :param param_name: The name of the parameter that was varied during the
                       benchmark (e.g., "Number of Rays").
    :type param_name: str
    :param output_dir: The directory where the plots will be saved.
    :type output_dir: str
    :param param_is_log: Whether to use a logarithmic scale for the x-axis.
    :type param_is_log: bool, optional
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Please install it to generate plots: pip install matplotlib")
        return

    os.makedirs(output_dir, exist_ok=True)

    engine_names = list(results.keys())

    # --- Plot Runtime ---
    plt.figure(figsize=(10, 6))
    for engine in engine_names:
        param_values = sorted(results[engine].keys())
        runtimes = [results[engine][p]['runtime_s'] for p in param_values]
        plt.plot(param_values, runtimes, marker='o', linestyle='-', label=engine)

    plt.title(f'Runtime vs. {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Runtime (seconds)')
    if param_is_log:
        plt.xscale('log')
    plt.grid(True)
    plt.legend()
    runtime_plot_path = os.path.join(output_dir, f'runtime_vs_{param_name.lower().replace(" ", "_")}.png')
    plt.savefig(runtime_plot_path)
    plt.close()
    print(f"Saved runtime plot to {runtime_plot_path}")

    # --- Plot Memory Usage ---
    plt.figure(figsize=(10, 6))
    for engine in engine_names:
        param_values = sorted(results[engine].keys())
        memories = [results[engine][p]['peak_memory_mb'] for p in param_values]
        plt.plot(param_values, memories, marker='o', linestyle='-', label=engine)

    plt.title(f'Peak Memory Usage vs. {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Peak Memory (MB)')
    if param_is_log:
        plt.xscale('log')
    plt.grid(True)
    plt.legend()
    memory_plot_path = os.path.join(output_dir, f'memory_vs_{param_name.lower().replace(" ", "_")}.png')
    plt.savefig(memory_plot_path)
    plt.close()
    print(f"Saved memory plot to {memory_plot_path}")
