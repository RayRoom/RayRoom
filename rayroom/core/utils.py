import numpy as np


def generate_rir(histogram, fs=44100, duration=2.0, random_phase=True):
    """Generates a Room Impulse Response (RIR) from a time-energy histogram.

    This function converts a list of reflection arrival times and their
    corresponding amplitudes into a discrete-time RIR signal. This is a
    common way to synthesize an RIR from the output of a ray tracing or
    image source model simulation.

    Responsibilities:
      * Convert a time-based energy histogram to a sampled RIR.
      * Handle the quantization of arrival times to sample indices.
      * Optionally apply random phase to simulate diffuse reflections.
      * Ensure the RIR has the specified duration and sample rate.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            # A simple histogram of reflections: (time_in_seconds, amplitude)
            reflection_histogram = [
                (0.01, 1.0),   # Direct sound
                (0.05, 0.5),   # First reflection
                (0.08, 0.3),   # Second reflection
                (0.08, 0.25)   # Another reflection at the same time
            ]
            
            fs = 44100
            rir_duration = 0.2  # seconds
            
            rir = rt.core.utils.generate_rir(
                reflection_histogram, fs=fs, duration=rir_duration
            )
            
            # The result is a NumPy array representing the RIR

    :param histogram: A list of tuples, where each tuple contains the arrival
                      time (in seconds) and amplitude of a reflection.
    :type histogram: list[tuple[float, float]]
    :param fs: The sampling frequency in Hertz (Hz). Defaults to 44100.
    :type fs: int, optional
    :param duration: The desired duration of the RIR in seconds. Defaults to 2.0.
    :type duration: float, optional
    :param random_phase: If `True`, applies a random sign flip to each
                         reflection to simulate phase variations from diffuse
                         surfaces. Defaults to `True`.
    :type random_phase: bool, optional
    :return: The generated Room Impulse Response.
    :rtype: np.ndarray
    """
    if not histogram:
        return np.zeros(int(fs * duration))

    # Sort by time
    histogram.sort(key=lambda x: x[0])

    times = np.array([t for t, a in histogram])
    amplitudes = np.array([a for t, a in histogram])

    # Discard late reflections
    valid_indices = times < duration
    times = times[valid_indices]
    amplitudes = amplitudes[valid_indices]

    if len(times) == 0:
        return np.zeros(int(fs * duration))

    if random_phase:
        # Apply random sign flips to break phase coherence for diffuse sounds
        signs = np.random.choice([-1, 1], size=len(amplitudes))
        amplitudes *= signs

    # Create RIR
    rir_len = int(fs * duration)
    rir = np.zeros(rir_len)

    # Place amplitudes in the RIR
    indices = (times * fs).astype(int)

    # Handle multiple arrivals in the same sample bin
    np.add.at(rir, indices, amplitudes)

    return rir
