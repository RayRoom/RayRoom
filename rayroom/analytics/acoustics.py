import numpy as np
from scipy.signal import butter, lfilter


def schroeder_integration(rir):
    """Computes the Schroeder integral of a room impulse response (RIR).

    This function calculates the reverse cumulative squared sum of the RIR,
    which is used to analyze the decay characteristics of a room. The resulting
    decay curve is represented in decibels (dB).

    Responsibilities:
      * Calculate the energy of the RIR.
      * Compute the Schroeder integral.
      * Convert the result to a logarithmic scale (dB).

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt
            
            fs = 44100
            rir_duration = 2.0  # seconds
            num_samples = int(fs * rir_duration)
            
            # Generate a dummy RIR with exponential decay
            t = np.linspace(0, rir_duration, num_samples)
            rir = np.exp(-5 * t) * np.random.randn(num_samples)
            
            # Compute the Schroeder curve
            sch_db = rt.analytics.acoustics.schroeder_integration(rir)
            
            # The sch_db can now be used to calculate RT60, EDT, etc.


    :param rir: The room impulse response signal.
    :type rir: np.ndarray
    :return: The Schroeder decay curve in dB.
    :rtype: np.ndarray
    """
    # Use 64-bit floats to avoid overflow with squared values
    energy = np.power(rir.astype(np.float64), 2)
    sch = np.cumsum(energy[::-1])[::-1]

    max_sch = np.max(sch)
    if max_sch < 1e-20:  # If signal is essentially silent
        return np.full_like(sch, -200.0)

    sch_db = 10 * np.log10(sch / max_sch + 1e-20)
    return sch_db


def _calculate_decay_time(sch_db, fs, t_start=None, t_end=None):
    """Calculates the reverberation time (RT60) from a Schroeder decay curve.

    This is a helper function that performs a linear regression on a specified
    portion of the Schroeder curve to determine the time it takes for the
    sound level to decay by 60 dB. It can be adapted for T20, T30, or EDT
    by changing the start and end points of the regression.

    Responsibilities:
      * Identify the region of the decay curve for linear regression.
      * Perform a linear fit to find the decay slope.
      * Calculate the RT60 based on the slope.
      * Handle cases with insufficient decay or low dynamic range.

    :param sch_db: Schroeder decay curve in dB.
    :type sch_db: np.ndarray
    :param fs: Sampling frequency in Hz.
    :type fs: int
    :param t_start: Start level for the linear fit (in dB, e.g., -5).
    :type t_start: float, optional
    :param t_end: End level for the linear fit (in dB, e.g., -25 for T20).
    :type t_end: float, optional
    :return: The calculated RT60 value in seconds.
    :rtype: float
    """
    if t_start is None:
        t_start = -5
    if t_end is None:
        t_end = -25

    # Find start of decay (first point under max)
    try:
        start_idx = np.where(sch_db < sch_db[0] - 1e-6)[0][0]
    except IndexError:
        start_idx = 0  # Flat curve

    # Find points for the linear fit
    try:
        fit_start_idx = np.where(sch_db[start_idx:] <= t_start)[0][0] + start_idx
        fit_end_idx = np.where(sch_db[fit_start_idx:] <= t_end)[0][0] + fit_start_idx
    except IndexError:
        # Not enough decay, try to fit on a smaller range
        try:
            fit_start_idx = np.where(sch_db <= -5)[0][0]
            # Find the point where it has decayed at least 15 dB more
            end_val = sch_db[fit_start_idx] - 15
            fit_end_idx = np.where(sch_db[fit_start_idx:] <= end_val)[0][0] + fit_start_idx
        except IndexError:
            return np.nan

    if fit_end_idx <= fit_start_idx + 10:  # Need at least a few points
        return np.nan

    # Time vector for the selected range
    t = np.arange(fit_start_idx, fit_end_idx) / fs

    # Linear regression
    coeffs = np.polyfit(t, sch_db[fit_start_idx:fit_end_idx], 1)
    slope = coeffs[0]

    # RT60 is the time to decay by 60 dB
    if slope >= -1e-3:  # Effectively zero or positive slope
        return np.nan

    rt60 = -60 / slope
    return rt60


def calculate_rt60(sch_db, fs):
    """Calculates T20 reverberation time from a Schroeder decay curve.

    This function estimates the RT60 by performing a linear regression on the
    decay curve between -5 dB and -25 dB. The resulting slope is then
    extrapolated to find the time for a 60 dB decay.

    Responsibilities:
      * Utilize `_calculate_decay_time` to compute T20.
      * Provide a simplified interface for a common acoustic parameter.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            fs = 44100
            rir_duration = 2.0
            num_samples = int(fs * rir_duration)
            t = np.linspace(0, rir_duration, num_samples)
            rir = np.exp(-5 * t) * np.random.randn(num_samples)
            
            sch_db = rt.analytics.acoustics.schroeder_integration(rir)
            t20 = rt.analytics.acoustics.calculate_rt60(sch_db, fs)
            
            print(f"T20 Reverberation Time: {t20:.2f} seconds")

    :param sch_db: Schroeder decay curve in dB.
    :type sch_db: np.ndarray
    :param fs: Sampling frequency in Hz.
    :type fs: int
    :return: The calculated T20 value in seconds.
    :rtype: float
    """
    return _calculate_decay_time(sch_db, fs, t_start=-5, t_end=-25)


def calculate_edt(sch_db, fs):
    """Calculates Early Decay Time (EDT) from a Schroeder decay curve.

    EDT is calculated similarly to RT60, but the linear regression is
    performed on the initial part of the decay curve, from 0 dB to -10 dB.
    This metric is often more correlated with the perceived reverberation.

    Responsibilities:
      * Utilize `_calculate_decay_time` to compute EDT.
      * Focus on the early part of the sound decay.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            fs = 44100
            rir_duration = 2.0
            num_samples = int(fs * rir_duration)
            t = np.linspace(0, rir_duration, num_samples)
            rir = np.exp(-5 * t) * np.random.randn(num_samples)
            
            sch_db = rt.analytics.acoustics.schroeder_integration(rir)
            edt = rt.analytics.acoustics.calculate_edt(sch_db, fs)
            
            print(f"Early Decay Time: {edt:.2f} seconds")

    :param sch_db: Schroeder decay curve in dB.
    :type sch_db: np.ndarray
    :param fs: Sampling frequency in Hz.
    :type fs: int
    :return: The calculated EDT value in seconds.
    :rtype: float
    """
    return _calculate_decay_time(sch_db, fs, t_start=0, t_end=-10)


def octave_band_filter(data, fs, center_freq, order=4):
    """Filters a signal into a specific octave band.

    This function applies a Butterworth band-pass filter to isolate a
    frequency range defined by a center frequency. The band is one octave
    wide.

    Responsibilities:
      * Design a Butterworth filter for the specified octave band.
      * Apply the filter to the input signal.
      * Handle frequency normalization and boundary conditions.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            fs = 44100
            duration = 1.0
            num_samples = int(fs * duration)
            t = np.linspace(0, duration, num_samples, endpoint=False)
            
            # White noise signal
            noise = np.random.randn(num_samples)
            
            # Filter the noise in the 1000 Hz octave band
            filtered_noise = rt.analytics.acoustics.octave_band_filter(
                noise, fs, center_freq=1000
            )

    :param data: Input signal.
    :type data: np.ndarray
    :param fs: Sampling frequency.
    :type fs: int
    :param center_freq: Center frequency of the octave band.
    :type center_freq: float
    :param order: Order of the Butterworth filter.
    :type order: int, optional
    :return: The filtered signal.
    :rtype: np.ndarray
    """
    # Octave band limits (factor of sqrt(2))
    f_low = center_freq / np.sqrt(2)
    f_high = center_freq * np.sqrt(2)
    # Nyquist frequency
    nyquist = 0.5 * fs
    # Critical frequencies (normalized)
    low = f_low / nyquist
    high = f_high / nyquist
    # Avoid issues at boundaries
    if high >= 1.0:
        high = 0.9999
    if low <= 0:
        low = 1e-6
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def get_octave_bands(subdivisions=1):
    """Returns standard octave band center frequencies.

    Provides a list of center frequencies for standard octave bands used in
    acoustic analysis. Optionally, it can generate intermediate frequencies
    for finer resolution.

    Responsibilities:
      * Define standard octave band center frequencies.
      * Generate subdivided frequencies if requested.

    Example:

        .. code-block:: python

            import rayroom as rt

            # Standard octave bands
            bands = rt.analytics.acoustics.get_octave_bands()
            print("Standard bands:", bands)
            
            # 1/3-octave bands (approximately)
            sub_bands = rt.analytics.acoustics.get_octave_bands(subdivisions=3)
            print("1/3-octave bands:", sub_bands)

    :param subdivisions: Number of points per octave interval.
                         For example, 1 for standard bands, 3 for 1/3-octave bands.
    :type subdivisions: int
    :return: An array of octave band center frequencies.
    :rtype: np.ndarray
    """
    base_bands = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    if subdivisions <= 1:
        return np.round(base_bands).astype(int)

    all_bands = []
    for i in range(len(base_bands) - 1):
        start_freq = base_bands[i]
        end_freq = base_bands[i+1]
        # Use endpoint=False to avoid duplicating frequencies at interval boundaries
        sub_bands = np.geomspace(start_freq, end_freq, subdivisions, endpoint=False)
        all_bands.extend(sub_bands)

    all_bands.append(base_bands[-1])  # Manually add the last band center

    return np.round(all_bands).astype(int)


def calculate_clarity(rir, fs, time_threshold_ms):
    """Calculates a clarity metric (e.g., C50, C80) from an RIR.

    Clarity metrics quantify the ratio of early-arriving sound energy to
    late-arriving sound energy. C50 and C80 are common metrics for speech
    and music, respectively.

    Responsibilities:
      * Separate the RIR into early and late energy components.
      * Calculate the ratio of early to late energy.
      * Express the result in decibels.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            fs = 44100
            rir_duration = 1.0
            num_samples = int(fs * rir_duration)
            t = np.linspace(0, rir_duration, num_samples)
            rir = np.exp(-6 * t) * np.random.randn(num_samples)
            
            # Calculate C50 (clarity for speech)
            c50 = rt.analytics.acoustics.calculate_clarity(rir, fs, 50)
            print(f"C50: {c50:.2f} dB")
            
            # Calculate C80 (clarity for music)
            c80 = rt.analytics.acoustics.calculate_clarity(rir, fs, 80)
            print(f"C80: {c80:.2f} dB")

    :param rir: The room impulse response.
    :type rir: np.ndarray
    :param fs: Sampling frequency.
    :type fs: int
    :param time_threshold_ms: The time threshold in milliseconds (50 for C50, 80 for C80).
    :type time_threshold_ms: float
    :return: Clarity value in dB.
    :rtype: float
    """
    is_ambisonic = rir.ndim > 1
    rir_to_process = rir[:, 0] if is_ambisonic else rir

    # Find the index of the direct sound (maximum of the RIR)
    direct_sound_idx = np.argmax(np.abs(rir_to_process))

    # Convert time threshold to samples
    threshold_samples = int((time_threshold_ms / 1000.0) * fs)

    # Early energy: from direct sound up to the threshold
    early_energy_end_idx = direct_sound_idx + threshold_samples
    early_energy = np.sum(rir_to_process[direct_sound_idx:early_energy_end_idx]**2)

    # Late energy: from the threshold to the end of the RIR
    late_energy = np.sum(rir_to_process[early_energy_end_idx:]**2)

    if late_energy < 1e-20:  # Avoid division by zero if there's no late energy
        return 100.0  # Return a very high dB value

    clarity = 10 * np.log10(early_energy / late_energy)
    return clarity


def calculate_drr(rir, fs, direct_sound_window_ms=5):
    """Calculates the Direct-to-Reverberant Ratio (DRR).

    DRR is the ratio of the energy of the direct sound to the energy of the
    reverberant sound. It is a measure of the relative amount of direct
    sound compared to reflected sound at a listening position.

    Responsibilities:
      * Identify the direct sound component within the RIR.
      * Separate direct and reverberant energy.
      * Calculate the ratio and express it in decibels.

    Example:

        .. code-block:: python

            import numpy as np
            import rayroom as rt

            fs = 44100
            rir_duration = 1.0
            num_samples = int(fs * rir_duration)
            t = np.linspace(0, rir_duration, num_samples)
            rir = np.exp(-6 * t) * np.random.randn(num_samples)
            
            # Calculate DRR
            drr = rt.analytics.acoustics.calculate_drr(rir, fs)
            print(f"DRR: {drr:.2f} dB")

    :param rir: The room impulse response.
    :type rir: np.ndarray
    :param fs: Sampling frequency.
    :type fs: int
    :param direct_sound_window_ms: The window size in ms to consider as direct sound.
    :type direct_sound_window_ms: float
    :return: DRR value in dB.
    :rtype: float
    """
    is_ambisonic = rir.ndim > 1
    rir_to_process = rir[:, 0] if is_ambisonic else rir

    # Find the index of the direct sound (maximum of the RIR)
    direct_sound_idx = np.argmax(np.abs(rir_to_process))

    # Define window for direct sound in samples
    window_samples = int((direct_sound_window_ms / 1000.0) * fs)
    direct_energy_start = max(0, direct_sound_idx - window_samples // 2)
    direct_energy_end = min(len(rir_to_process), direct_sound_idx + window_samples // 2)

    direct_energy = np.sum(rir_to_process[direct_energy_start:direct_energy_end]**2)

    # Reverberant energy is everything else
    reverberant_energy = np.sum(rir_to_process**2) - direct_energy

    if reverberant_energy < 1e-20:
        return 100.0  # Very high dB if no reverberant energy

    drr = 10 * np.log10(direct_energy / reverberant_energy)
    return drr
